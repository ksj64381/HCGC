#!/usr/bin/env python
"""
experiments.py -- HCGC accuracy-compression tradeoff sweep.

Runs benchmark.py's compress?뭪rain pipeline over a list of retention ratios
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
from pathlib import Path
import numpy as np

from benchmark import (
    LOADERS,
    _COMPRESSORS,
    _DOWNSTREAM_MODELS,
    _add_degree_features,
    _FULL_BATCH_NODE_LIMIT,
    run_baseline,
    run_once,
)


# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
# Sweep runner
# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧

def run_sweep(dataset, ratios, runs=3, warmup=1, device='auto', root='data',
              pretrain=True, train_epochs=200, train_hidden=256,
              mini_batch_size=512, model_name='sage', baseline=True,
              baseline_stats=None,
              emb_method='gnn', coarsen_l2_normalize=True,
              relprop_hops=2, relprop_outdim=128,
              type_thresholds=False, metapath_thresholds=False,
              edge_weight_mode='binary', compressor='hcgc',
              ratio_search='fast', auto_search_runs=8,
              auto_target_tolerance=None):
    """Run compress?뭪rain for each ratio and return collected stats.

    Returns
    -------
    base_stats : dict | None   (mean/std test_acc and t_train of baseline)
    sweep      : list[dict]    one entry per ratio, with mean/std of all metrics
    """
    W = 92
    print(f"\n{'='*W}")
    print(f"  HCGC Ratio Sweep")
    print(f"{'='*W}")
    print(f"  dataset  : {dataset}")
    print(f"  ratios   : {ratios}")
    print(f"  compressor: {compressor} ({_COMPRESSORS[compressor]})")
    print(f"  search   : {ratio_search}")
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

    # ?? Load dataset ??????????????????????????????????????????????????????????
    print(f"Loading {dataset} ...")
    data, target_type = LOADERS[dataset](root)
    data = _add_degree_features(data)

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"  node types : {list(data.node_types)}")
    print(f"  total nodes: {n_nodes:,}   total edges: {n_edges:,}")
    print(f"  target type: {target_type!r}")
    if n_nodes > _FULL_BATCH_NODE_LIMIT:
        print(f"  [large graph -mini-batch downstream (batch_size={mini_batch_size})]")

    # ?? Baseline ??????????????????????????????????????????????????????????????
    base_stats = baseline_stats
    if base_stats is not None:
        print(f"\nBaseline: reusing original-graph result "
              f"acc={base_stats['acc_mean']:.4f} +/- {base_stats['acc_std']:.4f} "
              f"t={base_stats['t_mean']:.1f}s")
    elif baseline:
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
        print(f"  Baseline  acc={base_stats['acc_mean']:.4f} +/- {base_stats['acc_std']:.4f}"
              f"  t={base_stats['t_mean']:.1f}s")

    # ?? Warmup (once, for the first ratio) ???????????????????????????????????
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
                     edge_weight_mode=edge_weight_mode,
                     compressor=compressor,
                     ratio_search=ratio_search,
                     auto_search_runs=auto_search_runs,
                     auto_target_tolerance=auto_target_tolerance)
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
                     edge_weight_mode=edge_weight_mode,
                     compressor=compressor,
                     ratio_search=ratio_search,
                     auto_search_runs=auto_search_runs,
                     auto_target_tolerance=auto_target_tolerance)
            print(f"  warmup {i+1}/{warmup} [pretrain={pretrain}]  ({time.perf_counter()-t_wu:.1f}s)")

    # ?? Ratio sweep ???????????????????????????????????????????????????????????
    sweep = []
    for ratio in ratios:
        print(f"\nRatio {ratio:.2f}  ({1/ratio:.1f}x target)  - {runs} run(s) ...")
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
                compressor      = compressor,
                ratio_search    = ratio_search,
                auto_search_runs = auto_search_runs,
                auto_target_tolerance = auto_target_tolerance,
            )
            recs.append(r)
            print(f"comp={r['compression']:.2f}x  "
                  f"t_total={r['t_total']:.1f}s  "
                  f"test_acc={r['test_acc']:.4f}  "
                  f"val_oracle={r.get('oracle_val_acc', float('nan')):.4f}  "
                  f"emb_d={r.get('target_emb_distortion', float('nan')):.4f}")

        def _s(key):
            v = np.array([r.get(key, float('nan')) for r in recs], dtype=float)
            if np.isnan(v).all():
                return float('nan'), float('nan')
            return float(np.nanmean(v)), float(np.nanstd(v))

        acc_m,  acc_s  = _s('test_acc')
        vo_m,   vo_s   = _s('oracle_val_acc')
        to_m,   to_s   = _s('oracle_acc')
        og_m,   og_s   = _s('oracle_gap')
        ed_m,   ed_s   = _s('target_emb_distortion')
        ec_m,   ec_s   = _s('target_emb_cosine')
        comp_m, comp_s = _s('compression')
        tt_m,   tt_s   = _s('t_total')
        tc_m,   tc_s   = _s('t_compress')
        tr_m,   tr_s   = _s('t_train')

        entry = {
            'ratio':    ratio,
            'acc_mean': acc_m, 'acc_std': acc_s,
            'val_oracle_mean': vo_m, 'val_oracle_std': vo_s,
            'oracle_mean': to_m, 'oracle_std': to_s,
            'oracle_gap_mean': og_m, 'oracle_gap_std': og_s,
            'emb_dist_mean': ed_m, 'emb_dist_std': ed_s,
            'emb_cos_mean': ec_m, 'emb_cos_std': ec_s,
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


# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
# Summary table
# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧

def print_sweep_table(base_stats, sweep, dataset):
    W = 68
    print(f"\n{'='*W}")
    print(f"  SWEEP RESULTS   dataset={dataset}   ({len(sweep)} ratios)")
    print(f"{'='*W}")

    has_base = base_stats is not None

    # header
    if has_base:
        print(f"  {'ratio':>6}  {'compr.':>7}  {'test_acc':>14}  "
              f"{'val_orcl':>8}  {'emb_d':>7}  {'drop':>7}  "
              f"{'speedup':>8}  {'t_total':>9}")
        print("  " + "-" * 88)
        print(f"  {'baseline':>6}  {'  1.00x':>7}  "
              f"{base_stats['acc_mean']:>7.4f}+/-{base_stats['acc_std']:.4f}  "
              f"{'n/a':>8}  {'n/a':>7}  {'n/a':>7}  {'  1.00x':>8}  "
              f"{base_stats['t_mean']:>8.1f}s")
        print("  " + "-" * 88)
    else:
        print(f"  {'ratio':>6}  {'compr.':>7}  {'test_acc':>14}  "
              f"{'val_orcl':>8}  {'emb_d':>7}  {'t_total':>9}")
        print("  " + "-" * 68)

    for e in sweep:
        acc_str  = f"{e['acc_mean']:.4f}+/-{e['acc_std']:.4f}"
        comp_str = f"{e['comp_mean']:5.2f}x"
        vo_str   = f"{e.get('val_oracle_mean', float('nan')):.4f}"
        ed_str   = f"{e.get('emb_dist_mean', float('nan')):.4f}"
        tt_str   = f"{e['tt_mean']:7.1f}s"
        if has_base:
            drop_str = f"{e['acc_drop']:+.4f}"
            su_str   = f"{e['train_speedup']:6.1f}x"
            print(f"  {e['ratio']:>6.2f}  {comp_str:>7}  {acc_str:>14}  "
                  f"{vo_str:>8}  {ed_str:>7}  {drop_str:>7}  "
                  f"{su_str:>8}  {tt_str:>9}")
        else:
            print(f"  {e['ratio']:>6.2f}  {comp_str:>7}  {acc_str:>14}  "
                  f"{vo_str:>8}  {ed_str:>7}  {tt_str:>9}")

    print(f"{'='*W}\n")


def save_sweep_plot(sweep, dataset, model_name, plot_dir, compressor='hcgc',
                    ratio_search='fast'):
    if not plot_dir:
        return None
    import json
    import csv
    import subprocess

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proxy_sweep_{dataset}_{model_name}_{compressor}_{ratio_search}"
    out_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    x = np.array([e['comp_mean'] for e in sweep], dtype=float)
    order = np.argsort(x)
    x = x[order]

    def arr(key):
        return np.array([sweep[i].get(key, float('nan')) for i in order], dtype=float)

    payload = {
        'x': x.tolist(),
        'acc': arr('acc_mean').tolist(),
        'val_oracle': arr('val_oracle_mean').tolist(),
        'test_oracle': arr('oracle_mean').tolist(),
        'emb_dist': arr('emb_dist_mean').tolist(),
        'title': (f'{dataset} {model_name} {compressor} '
                  f'{ratio_search} compression proxy sweep'),
        'out': str(out_path),
    }
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['compression', 'test_acc', 'val_oracle',
                         'test_oracle', 'emb_dist'])
        for row in zip(payload['x'], payload['acc'], payload['val_oracle'],
                       payload['test_oracle'], payload['emb_dist']):
            writer.writerow(row)
    print(f"  [plot] saved data {csv_path}")

    script = r"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = json.loads(sys.argv[1])
fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
ax1.plot(p['x'], p['acc'], marker='o', label='test accuracy')
ax1.plot(p['x'], p['val_oracle'], marker='s', label='val oracle')
ax1.plot(p['x'], p['test_oracle'], marker='^', alpha=0.65, label='test oracle')
ax1.set_xlabel('Actual node compression (x)')
ax1.set_ylabel('Accuracy / oracle bound')
ax1.grid(True, alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(p['x'], p['emb_dist'], color='tab:red', marker='x',
         linestyle='--', label='target emb distortion')
ax2.set_ylabel('Embedding distortion')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
fig.suptitle(p['title'])
fig.tight_layout()
fig.savefig(p['out'], dpi=180)
plt.close(fig)
"""
    try:
        proc = subprocess.run(
            [sys.executable, '-c', script, json.dumps(payload)],
            check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"  [plot] PNG skipped: matplotlib is unavailable or failed ({exc})")
        return csv_path
    print(f"  [plot] saved {out_path}")
    return out_path


def print_compressor_table(all_results, dataset, models, compressors):
    W = 104
    print(f"\n{'='*W}")
    print(f"  COMPRESSOR COMPARISON   dataset={dataset}")
    print(f"{'='*W}")
    print(f"  {'model':>8}  {'compressor':>12}  {'ratio':>6}  {'compr.':>7}  "
          f"{'test_acc':>14}  {'drop':>8}  {'val_orcl':>8}  "
          f"{'emb_d':>7}  {'t_total':>9}")
    print("  " + "-" * 100)

    for model_name in models:
        for compressor in compressors:
            base_stats, sweep = all_results[(model_name, compressor)]
            for e in sweep:
                acc = f"{e['acc_mean']:.4f}+/-{e['acc_std']:.4f}"
                drop = e.get('acc_drop', float('nan'))
                drop_str = 'n/a' if np.isnan(drop) else f"{drop:+.4f}"
                print(f"  {model_name:>8}  {compressor:>12}  "
                      f"{e['ratio']:>6.2f}  {e['comp_mean']:>6.2f}x  "
                      f"{acc:>14}  {drop_str:>8}  "
                      f"{e.get('val_oracle_mean', float('nan')):>8.4f}  "
                      f"{e.get('emb_dist_mean', float('nan')):>7.4f}  "
                      f"{e['tt_mean']:>8.1f}s")
    print(f"{'='*W}\n")


def save_comparison_outputs(all_results, dataset, models, compressors,
                            plot_dir, ratio_search='fast'):
    if not plot_dir:
        return None
    import csv
    import json
    import subprocess

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = '-'.join(models)
    comp_tag = '-'.join(compressors)
    stem = f"compressor_compare_{dataset}_{model_tag}_{comp_tag}_{ratio_search}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"

    rows = []
    for model_name in models:
        for compressor in compressors:
            base_stats, sweep = all_results[(model_name, compressor)]
            for e in sweep:
                row = {
                    'dataset': dataset,
                    'model': model_name,
                    'compressor': compressor,
                    'ratio_search': ratio_search,
                    'ratio': e['ratio'],
                    'compression': e['comp_mean'],
                    'test_acc_mean': e['acc_mean'],
                    'test_acc_std': e['acc_std'],
                    'acc_drop': e.get('acc_drop', float('nan')),
                    'val_oracle': e.get('val_oracle_mean', float('nan')),
                    'test_oracle': e.get('oracle_mean', float('nan')),
                    'emb_dist': e.get('emb_dist_mean', float('nan')),
                    'emb_cos': e.get('emb_cos_mean', float('nan')),
                    't_total': e['tt_mean'],
                    't_compress': e['tc_mean'],
                    't_train': e['tr_mean'],
                    'train_speedup': e.get('train_speedup', float('nan')),
                    'baseline_acc_mean': (
                        base_stats['acc_mean'] if base_stats is not None
                        else float('nan')),
                    'baseline_acc_std': (
                        base_stats['acc_std'] if base_stats is not None
                        else float('nan')),
                    'baseline_t_train': (
                        base_stats['t_mean'] if base_stats is not None
                        else float('nan')),
                }
                rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        'dataset': dataset,
        'models': models,
        'compressors': compressors,
        'ratio_search': ratio_search,
        'rows': rows,
    }
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(_json_clean(payload), f, indent=2)

    print(f"  [compare] saved table {csv_path}")
    print(f"  [compare] saved data  {json_path}")

    script = r"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = json.loads(sys.argv[1])
rows = p['rows']
fig, ax = plt.subplots(figsize=(9.2, 5.2))
for model in p['models']:
    for comp in p['compressors']:
        pts = [r for r in rows if r['model'] == model and r['compressor'] == comp]
        pts = [r for r in pts if r['compression'] is not None and r['test_acc_mean'] is not None]
        pts.sort(key=lambda r: r['compression'])
        if not pts:
            continue
        label = comp if len(p['models']) == 1 else f"{model}/{comp}"
        ax.plot([r['compression'] for r in pts],
                [r['test_acc_mean'] for r in pts],
                marker='o', label=label)
ax.set_xlabel('Actual node compression (x)')
ax.set_ylabel('Test accuracy')
ax.grid(True, alpha=0.25)
ax.legend(loc='best', fontsize=8)
fig.suptitle(f"{p['dataset']} compressor comparison")
fig.tight_layout()
fig.savefig(p['out'], dpi=180)
plt.close(fig)
"""
    plot_payload = _json_clean({
        'dataset': dataset,
        'models': models,
        'compressors': compressors,
        'ratio_search': ratio_search,
        'rows': rows,
        'out': str(png_path),
    })
    try:
        subprocess.run(
            [sys.executable, '-c', script, json.dumps(plot_payload)],
            check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"  [compare] PNG skipped: matplotlib is unavailable or failed ({exc})")
        return csv_path

    print(f"  [compare] saved plot  {png_path}")
    return png_path


def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧
# CLI
# ?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧

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
                        help='Retention ratios to sweep (high -low)')
    parser.add_argument('--compressor', default='hcgc',
                        choices=list(_COMPRESSORS),
                        help='Compression method to sweep.')
    parser.add_argument('--compressors', nargs='+', choices=list(_COMPRESSORS),
                        metavar='C',
                        help='One or more compression methods to compare. '
                             'Overrides --compressor.')
    parser.add_argument('--all-compressors', action='store_true',
                        help='Run all available compression methods in one job.')
    parser.add_argument('--ratio-search', default='fast',
                        choices=['fast', 'precise'],
                        help='Target-ratio search mode for hcgc/cgc_type.')
    parser.add_argument('--auto-search-runs', type=int, default=8,
                        help='Max interpolation runs for --ratio-search precise.')
    parser.add_argument('--auto-target-tolerance', type=float, default=None,
                        help='Relative compression error tolerance. Default: '
                             '0.05 for precise, 0.15 for fast.')
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
    parser.add_argument('--plot-dir', default=None,
                        help='If set, save accuracy/proxy sweep plots to this directory.')
    args = parser.parse_args()

    ratios   = sorted(args.ratios, reverse=True)   # high -low (less -more compressed)
    models   = args.models if args.models else [args.model]
    if args.all_compressors:
        compressors = list(_COMPRESSORS)
    elif args.compressors:
        compressors = args.compressors
    else:
        compressors = [args.compressor]
    do_base  = not args.no_baseline

    all_results = {}      # (model_name, compressor) -> (base_stats, sweep)
    baseline_by_model = {}
    combo_idx = 0
    total_combos = len(models) * len(compressors)

    for mname in models:
        for compressor in compressors:
            combo_idx += 1
            print(f"\n{'#'*72}")
            print(f"  Job {combo_idx}/{total_combos}: model={mname}  "
                  f"compressor={compressor}")
            print(f"{'#'*72}")

            base_override = baseline_by_model.get(mname)
            run_baseline_now = do_base and base_override is None
            warmup_now = args.warmup if combo_idx == 1 else 0

            base_stats, sweep = run_sweep(
                dataset         = args.dataset,
                ratios          = ratios,
                runs            = args.runs,
                warmup          = warmup_now,
                device          = args.device,
                root            = args.root,
                pretrain        = not args.no_pretrain,
                train_epochs    = args.train_epochs,
                train_hidden    = args.train_hidden,
                mini_batch_size = args.mini_batch_size,
                model_name      = mname,
                baseline        = run_baseline_now,
                baseline_stats  = base_override,
                emb_method      = args.emb_method,
                coarsen_l2_normalize = not args.raw_no_l2,
                relprop_hops    = args.relprop_hops,
                relprop_outdim  = args.relprop_outdim,
                type_thresholds = args.type_thresholds,
                metapath_thresholds = args.metapath_thresholds,
                edge_weight_mode = args.edge_weight_mode,
                compressor      = compressor,
                ratio_search    = args.ratio_search,
                auto_search_runs = args.auto_search_runs,
                auto_target_tolerance = args.auto_target_tolerance,
            )
            if do_base and mname not in baseline_by_model:
                baseline_by_model[mname] = base_stats
            print_sweep_table(base_stats, sweep, args.dataset)
            save_sweep_plot(sweep, args.dataset, mname, args.plot_dir,
                            compressor, args.ratio_search)
            all_results[(mname, compressor)] = (base_stats, sweep)

    print_compressor_table(all_results, args.dataset, models, compressors)
    save_comparison_outputs(all_results, args.dataset, models, compressors,
                            args.plot_dir, args.ratio_search)


if __name__ == '__main__':
    main()
