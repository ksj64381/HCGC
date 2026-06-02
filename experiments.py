#!/usr/bin/env python
"""
experiments.py -- HCGC accuracy-compression tradeoff sweep.

Runs benchmark.py's compress→train pipeline over a list of retention ratios
and prints a comparison table.  Keeps benchmark.py clean (single-ratio only).

Usage
-----
    # ratio sweep on IMDB (with baseline), single model
    python experiments.py --dataset imdb --ratios 0.5 0.4 0.3 0.25 0.2 0.15 0.1

    # multi-model sweep: one table per model, summary table at the end
    python experiments.py --dataset dblp --ratios 0.3 0.1 --models sage hgt gat rgcn

    # quick smoke test (1 run, no warmup)
    python experiments.py --dataset dblp --ratios 0.3 0.2 0.1 --runs 1 --warmup 0
"""

import argparse
import sys
import time
import warnings
import numpy as np

from benchmark import (
    LOADERS,
    _DOWNSTREAM_MODELS,
    _add_degree_features,
    _FULL_BATCH_NODE_LIMIT,
    run_baseline,
    run_once,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sweep runner
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(dataset, ratios, runs=3, warmup=1, device='auto', root='data',
              pretrain=True, train_epochs=200, train_hidden=256,
              mini_batch_size=512, model_name='sage', baseline=True,
              emb_method='gnn', coarsen_l2_normalize=True,
              relprop_hops=2, relprop_outdim=128,
              type_thresholds=False, metapath_thresholds=False,
              edge_weight_mode='binary'):
    """Run compress→train for each ratio and return collected stats.

    Returns
    -------
    base_stats : dict | None   (mean/std test_acc and t_train of baseline)
    sweep      : list[dict]    one entry per ratio, with mean/std of all metrics
    """
    W = 68
    print(f"\n{'='*W}")
    print(f"  HCGC Ratio Sweep")
    print(f"{'='*W}")
    print(f"  dataset  : {dataset}")
    print(f"  ratios   : {ratios}")
    print(f"  model    : {model_name}")
    print(f"  pretrain : {pretrain}")
    print(f"  emb      : {emb_method if pretrain else 'raw'}")
    print(f"  edge_w   : {edge_weight_mode}")
    _thresh_mode = ('metapath-auto' if metapath_thresholds
                    else 'type-auto' if type_thresholds else 'global')
    print(f"  thresh   : {_thresh_mode}")
    print(f"  device   : {device}")
    print(f"  runs     : {warmup} warmup  +  {runs} timed")
    print(f"{'='*W}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading {dataset} ...")
    data, target_type = LOADERS[dataset](root)
    data = _add_degree_features(data)

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"  node types : {list(data.node_types)}")
    print(f"  total nodes: {n_nodes:,}   total edges: {n_edges:,}")
    print(f"  target type: {target_type!r}")
    if n_nodes > _FULL_BATCH_NODE_LIMIT:
        print(f"  [large graph — mini-batch downstream (batch_size={mini_batch_size})]")

    # ── Baseline ──────────────────────────────────────────────────────────────
    base_stats = None
    if baseline:
        print(f"\nBaseline: training on original graph ({runs} runs) ...")
        base_recs = []
        for i in range(runs):
            print(f"  baseline run {i+1}/{runs} ... ", end='', flush=True)
            acc, t = run_baseline(data, target_type, device,
                                  train_epochs=train_epochs,
                                  train_hidden=train_hidden,
                                  mini_batch_size=mini_batch_size,
                                  model_name=model_name)
            base_recs.append({'test_acc': acc, 't_train': t})
            print(f"t={t:.1f}s  test_acc={acc:.4f}")
        base_stats = {
            'acc_mean': float(np.mean([r['test_acc'] for r in base_recs])),
            'acc_std':  float(np.std ([r['test_acc'] for r in base_recs])),
            't_mean':   float(np.mean([r['t_train']  for r in base_recs])),
        }
        print(f"  Baseline  acc={base_stats['acc_mean']:.4f} ± {base_stats['acc_std']:.4f}"
              f"  t={base_stats['t_mean']:.1f}s")

    # ── Warmup (once, for the first ratio) ───────────────────────────────────
    if warmup > 0:
        wup_ratio = ratios[0]
        print(f"\nWarmup ({warmup} run(s), ratio={wup_ratio}) ...")
        for i in range(warmup):
            t_wu = time.perf_counter()
            run_once(data, target_type, ratio=wup_ratio, device=device,
                     pretrain=False, verbose=False,
                     mini_batch_size=mini_batch_size, model_name=model_name,
                     coarsen_l2_normalize=coarsen_l2_normalize,
                     relprop_hops=relprop_hops,
                     relprop_outdim=relprop_outdim,
                     type_thresholds=type_thresholds,
                     metapath_thresholds=metapath_thresholds,
                     edge_weight_mode=edge_weight_mode)
            print(f"  warmup {i+1}/{warmup} [no-pretrain]  ({time.perf_counter()-t_wu:.1f}s)")
            t_wu = time.perf_counter()
            run_once(data, target_type, ratio=wup_ratio, device=device,
                     pretrain=pretrain, verbose=False,
                     train_epochs=train_epochs, train_hidden=train_hidden,
                     mini_batch_size=mini_batch_size, model_name=model_name,
                     emb_method=emb_method,
                     coarsen_l2_normalize=coarsen_l2_normalize,
                     relprop_hops=relprop_hops,
                     relprop_outdim=relprop_outdim,
                     type_thresholds=type_thresholds,
                     metapath_thresholds=metapath_thresholds,
                     edge_weight_mode=edge_weight_mode)
            print(f"  warmup {i+1}/{warmup} [pretrain={pretrain}]  ({time.perf_counter()-t_wu:.1f}s)")

    # ── Ratio sweep ───────────────────────────────────────────────────────────
    sweep = []
    for ratio in ratios:
        print(f"\nRatio {ratio:.2f}  ({1/ratio:.1f}x target)  — {runs} run(s) ...")
        recs = []
        for i in range(runs):
            print(f"  run {i+1}/{runs} ... ", end='', flush=True)
            r = run_once(
                data, target_type,
                ratio           = ratio,
                device          = device,
                pretrain        = pretrain,
                train_epochs    = train_epochs,
                train_hidden    = train_hidden,
                verbose         = False,
                mini_batch_size = mini_batch_size,
                model_name      = model_name,
                emb_method      = emb_method,
                coarsen_l2_normalize = coarsen_l2_normalize,
                relprop_hops    = relprop_hops,
                relprop_outdim  = relprop_outdim,
                type_thresholds = type_thresholds,
                metapath_thresholds = metapath_thresholds,
                edge_weight_mode = edge_weight_mode,
            )
            recs.append(r)
            print(f"comp={r['compression']:.2f}x  "
                  f"t_total={r['t_total']:.1f}s  "
                  f"test_acc={r['test_acc']:.4f}")

        def _s(key):
            v = [r[key] for r in recs]
            return float(np.mean(v)), float(np.std(v))

        acc_m,  acc_s  = _s('test_acc')
        comp_m, comp_s = _s('compression')
        tt_m,   tt_s   = _s('t_total')
        tc_m,   tc_s   = _s('t_compress')
        tr_m,   tr_s   = _s('t_train')

        entry = {
            'ratio':    ratio,
            'acc_mean': acc_m, 'acc_std': acc_s,
            'comp_mean': comp_m,
            'tt_mean':  tt_m,  'tt_std':  tt_s,
            'tc_mean':  tc_m,
            'tr_mean':  tr_m,
        }
        if base_stats is not None:
            entry['acc_drop']      = acc_m - base_stats['acc_mean']
            entry['train_speedup'] = base_stats['t_mean'] / max(tr_m, 1e-6)
        sweep.append(entry)

    return base_stats, sweep


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_sweep_table(base_stats, sweep, dataset):
    W = 68
    print(f"\n{'='*W}")
    print(f"  SWEEP RESULTS   dataset={dataset}   ({len(sweep)} ratios)")
    print(f"{'='*W}")

    has_base = base_stats is not None

    # header
    if has_base:
        print(f"  {'ratio':>6}  {'compr.':>7}  {'test_acc':>14}  "
              f"{'drop':>7}  {'speedup':>8}  {'t_total':>9}")
        print(f"  {'─'*6}  {'─'*7}  {'─'*14}  "
              f"{'─'*7}  {'─'*8}  {'─'*9}")
        print(f"  {'baseline':>6}  {'  1.00x':>7}  "
              f"{base_stats['acc_mean']:>7.4f}±{base_stats['acc_std']:.4f}  "
              f"{'—':>7}  {'  1.00x':>8}  "
              f"{base_stats['t_mean']:>8.1f}s")
        print(f"  {'─'*6}  {'─'*7}  {'─'*14}  "
              f"{'─'*7}  {'─'*8}  {'─'*9}")
    else:
        print(f"  {'ratio':>6}  {'compr.':>7}  {'test_acc':>14}  {'t_total':>9}")
        print(f"  {'─'*6}  {'─'*7}  {'─'*14}  {'─'*9}")

    for e in sweep:
        acc_str  = f"{e['acc_mean']:.4f}±{e['acc_std']:.4f}"
        comp_str = f"{e['comp_mean']:5.2f}x"
        tt_str   = f"{e['tt_mean']:7.1f}s"
        if has_base:
            drop_str = f"{e['acc_drop']:+.4f}"
            su_str   = f"{e['train_speedup']:6.1f}x"
            print(f"  {e['ratio']:>6.2f}  {comp_str:>7}  {acc_str:>14}  "
                  f"{drop_str:>7}  {su_str:>8}  {tt_str:>9}")
        else:
            print(f"  {e['ratio']:>6.2f}  {comp_str:>7}  {acc_str:>14}  {tt_str:>9}")

    print(f"{'='*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='HCGC accuracy-compression tradeoff sweep.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset',  required=True, choices=list(LOADERS),
                        help='Dataset to sweep')
    parser.add_argument('--ratios',   type=float, nargs='+',
                        default=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1],
                        metavar='R',
                        help='Retention ratios to sweep (high → low)')
    parser.add_argument('--runs',     type=int, default=3,
                        help='Timed runs per ratio')
    parser.add_argument('--warmup',   type=int, default=1,
                        help='Warmup runs before the first ratio (excluded from stats)')
    parser.add_argument('--device',   default='auto')
    parser.add_argument('--root',     default='data',
                        help='Dataset download root')
    parser.add_argument('--no-pretrain', action='store_true')
    parser.add_argument('--emb-method', default='gnn',
                        choices=['gnn', 'fast', 'relprop', 'metapath2vec'])
    parser.add_argument('--raw-no-l2', action='store_true',
                        help='Disable row-wise L2 normalization for raw-feature coarsening.')
    parser.add_argument('--relprop-hops', type=int, default=2)
    parser.add_argument('--relprop-outdim', type=int, default=128)
    parser.add_argument('--type-thresholds', action='store_true',
                        help='Use auto-calibrated per-source-type thresholds.')
    parser.add_argument('--metapath-thresholds', action='store_true',
                        help='Use auto-calibrated per-(source type, mediator type) '
                             'thresholds. Takes precedence over --type-thresholds.')
    parser.add_argument('--edge-weight-mode', default='binary',
                        choices=['binary', 'count', 'log_count', 'density'],
                        help='Compressed super-edge weighting mode.')
    parser.add_argument('--train-epochs', type=int, default=200)
    parser.add_argument('--train-hidden', type=int, default=256)
    parser.add_argument('--mini-batch-size', type=int, default=512)
    parser.add_argument('--model',    default='sage', choices=list(_DOWNSTREAM_MODELS),
                        help='Downstream GNN architecture (ignored when --models is used)')
    parser.add_argument('--models',   nargs='+', choices=list(_DOWNSTREAM_MODELS),
                        metavar='M',
                        help='One or more GNN models to compare. Overrides --model. '
                             'Runs a sweep for each model; prints a cross-model summary table. '
                             f'Choices: {list(_DOWNSTREAM_MODELS)}')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip original-graph baseline training')
    args = parser.parse_args()

    ratios   = sorted(args.ratios, reverse=True)   # high → low (less → more compressed)
    models   = args.models if args.models else [args.model]
    do_base  = not args.no_baseline

    if len(models) == 1:
        # ── Single-model path (original behaviour) ────────────────────────────
        base_stats, sweep = run_sweep(
            dataset         = args.dataset,
            ratios          = ratios,
            runs            = args.runs,
            warmup          = args.warmup,
            device          = args.device,
            root            = args.root,
            pretrain        = not args.no_pretrain,
            train_epochs    = args.train_epochs,
            train_hidden    = args.train_hidden,
            mini_batch_size = args.mini_batch_size,
            model_name      = models[0],
            baseline        = do_base,
            emb_method      = args.emb_method,
            coarsen_l2_normalize = not args.raw_no_l2,
            relprop_hops    = args.relprop_hops,
            relprop_outdim  = args.relprop_outdim,
            type_thresholds = args.type_thresholds,
            metapath_thresholds = args.metapath_thresholds,
            edge_weight_mode = args.edge_weight_mode,
        )
        print_sweep_table(base_stats, sweep, args.dataset)

    else:
        # ── Multi-model path ──────────────────────────────────────────────────
        # Collect results per model, then print a cross-model summary table.
        # Baseline is measured ONCE per model (it also depends on the model arch).
        all_results = {}   # model_name -> (base_stats, sweep)

        for mi, mname in enumerate(models):
            print(f"\n{'#'*68}")
            print(f"  Model {mi+1}/{len(models)}: {mname.upper()}")
            print(f"{'#'*68}")
            base_stats, sweep = run_sweep(
                dataset         = args.dataset,
                ratios          = ratios,
                runs            = args.runs,
                warmup          = args.warmup if mi == 0 else 0,  # warmup once
                device          = args.device,
                root            = args.root,
                pretrain        = not args.no_pretrain,
                train_epochs    = args.train_epochs,
                train_hidden    = args.train_hidden,
                mini_batch_size = args.mini_batch_size,
                model_name      = mname,
                baseline        = do_base,
                emb_method      = args.emb_method,
                coarsen_l2_normalize = not args.raw_no_l2,
                relprop_hops    = args.relprop_hops,
                relprop_outdim  = args.relprop_outdim,
                type_thresholds = args.type_thresholds,
                metapath_thresholds = args.metapath_thresholds,
                edge_weight_mode = args.edge_weight_mode,
            )
            print_sweep_table(base_stats, sweep, args.dataset)
            all_results[mname] = (base_stats, sweep)

        # ── Cross-model summary ───────────────────────────────────────────────
        W = 72
        print(f"\n{'='*W}")
        print(f"  MULTI-MODEL SUMMARY   dataset={args.dataset}   ratios={ratios}")
        print(f"{'='*W}")

        # Header row: model names
        ratio_col_w = 8
        model_col_w = 16   # "0.XXXX±0.XXXX" = 13 chars + padding
        header = f"  {'ratio':>{ratio_col_w}}"
        for mname in models:
            header += f"  {mname.upper():^{model_col_w}}"
        print(header)

        # Baseline row
        bline_row = f"  {'baseline':>{ratio_col_w}}"
        for mname in models:
            bs, _ = all_results[mname]
            if bs is not None:
                bline_row += f"  {bs['acc_mean']:6.4f}±{bs['acc_std']:.4f}  "
            else:
                bline_row += f"  {'—':^{model_col_w}}"
        print(bline_row)
        print(f"  {'─'*ratio_col_w}" + f"  {'─'*model_col_w}" * len(models))

        # One row per ratio
        for ri, ratio in enumerate(ratios):
            row = f"  {ratio:>{ratio_col_w}.2f}"
            for mname in models:
                _, sweep = all_results[mname]
                entry = sweep[ri]
                row += f"  {entry['acc_mean']:6.4f}±{entry['acc_std']:.4f}  "
            print(row)

        print(f"{'='*W}\n")


if __name__ == '__main__':
    main()
