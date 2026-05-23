"""
hcgc/_pipeline.py -- High-level pipeline: eval_pipeline and phase-split API
                     (_load_and_pretrain, _coarsen_from_context, etc.).
"""

import time
import numpy as np
import torch

import hcgc._config as _cfg
from hcgc._config import _CFG, set_seed
from hcgc._model import (
    build_model, _train_auto, _extract_emb_auto, _is_large_graph,
    extract_metapath2vec_embeddings, extract_emb_flat_arrays,
    eval_compressed_on_original, MINI_BATCH_THRESHOLD,
    train_full_batch, fast_embed_hetero,
)
from hcgc._coarsen import (
    extract_flat_arrays, build_compressed_data,
    _run_coarsen, auto_coarsen,
    _probe_from_coalition_map, _baseline_probe,
)


# ══════════════════════════════════════════════════════════════════════════════
# Eval pipeline
# ══════════════════════════════════════════════════════════════════════════════

def eval_pipeline(label, coalition_map, data, offsets, type_boundaries,
                  coarsen_time, args, trial_seed=None, emb_dict=None):
    """Train + evaluate on a compressed graph; return a result dict."""
    total_orig = len(coalition_map)
    total_comp = len(np.unique(coalition_map))
    compression = total_orig / total_comp

    per_type = {}
    for i, nt in enumerate(_CFG.node_types):
        s = offsets[nt]; e = int(type_boundaries[i])
        no = e - s; nc = len(np.unique(coalition_map[s:e]))
        per_type[nt] = {'orig': no, 'comp': nc, 'ratio': nc / no}

    print(f"\n  [{label}] {total_orig:,} -> {total_comp:,} nodes "
          f"({(1-total_comp/total_orig)*100:.1f}% reduced, {compression:.2f}x)")
    for nt, v in per_type.items():
        print(f"    {nt:14s}: {v['orig']:6,} -> {v['comp']:5,} ({v['ratio']*100:.1f}%)")

    comp_data, local_cm, comp_stats = build_compressed_data(
        data, coalition_map, offsets, type_boundaries,
        use_soft_labels=args.use_soft_labels,
        emb_dict=emb_dict, emb_temp=args.emb_temp)

    _probe_ctx = {
        'data':            data,
        'offsets':         offsets,
        'type_boundaries': type_boundaries,
        'emb_dict':        emb_dict or {},
    }
    probe_acc  = _probe_from_coalition_map(coalition_map, _probe_ctx)
    probe_base = _baseline_probe(_probe_ctx)
    probe_loss = (probe_base - probe_acc) if (not np.isnan(probe_acc) and not np.isnan(probe_base)) else float('nan')
    if not np.isnan(probe_acc):
        print(f"  [{label}] probe_acc={probe_acc:.4f}  "
              f"baseline={probe_base:.4f}  loss={probe_loss:+.4f}")
    else:
        print(f"  [{label}] probe_acc=N/A (embeddings not available)")

    print(f"\n  [{label}] Training on compressed graph ...")
    if trial_seed is not None:
        set_seed(trial_seed)
    model = build_model(comp_data, args.hidden, args.dropout, args.num_layers,
                        gnn_model=getattr(args, 'gnn_model', 'sage'))
    t_tr  = time.time()
    result, trained = _train_auto(model, comp_data,
                                  epochs=args.epochs, lr=args.lr,
                                  args=args, desc=label)
    t_tr = time.time() - t_tr

    acc_orig = eval_compressed_on_original(
        trained, comp_data.to(_cfg.DEVICE), local_cm[_CFG.target_type], data)
    print(f"  [{label}] -> original test acc: {acc_orig:.4f}")
    del model, trained
    if _cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    edge_ratio = comp_stats['edge_ratio']
    return {
        'method':           label,
        'compression':      round(compression, 4),
        'pct_reduced':      round((1 - total_comp / total_orig) * 100, 2),
        'edge_compression': round(edge_ratio, 4),
        'edge_pct_reduced': round((1 - 1 / max(edge_ratio, 1e-6)) * 100, 2),
        'edges_orig':       comp_stats['edges_orig'],
        'edges_comp':       comp_stats['edges_comp'],
        'probe_acc':        round(probe_acc, 4) if not np.isnan(probe_acc) else float('nan'),
        'probe_base':       round(probe_base, 4) if not np.isnan(probe_base) else float('nan'),
        'probe_loss':       round(probe_loss, 4) if not np.isnan(probe_loss) else float('nan'),
        'val_acc':          round(result['val'], 4),
        'test_acc_super':   round(result['test'], 4),
        'test_acc_orig':    round(acc_orig, 4),
        'best_epoch':       result['epoch'],
        'coarsen_time':     round(coarsen_time, 2),
        'train_time':       round(t_tr, 1),
        'per_type':         per_type,
    }


def _aggregate(trial_list, key):
    vals = [t[key] for t in trial_list if key in t]
    if not vals: return float('nan'), 0.0
    arr = np.array(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


# ══════════════════════════════════════════════════════════════════════════════
# Phase-split API  (pretrain once, coarsen + train separately)
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_pretrain(data, args):
    """Optionally pretrain GNN for embeddings on an already-loaded HeteroData.

    Args:
        data: PyG HeteroData (already loaded by the caller)
        args: argument namespace

    Returns a context dict reusable across multiple coarsening runs.
    """
    src_nodes, dst_nodes, weights, all_features, type_boundaries, feature_dims, offsets = \
        extract_flat_arrays(data)

    emb_dict = None
    coarsen_features, coarsen_feat_dims = all_features, feature_dims

    emb_method = getattr(args, 'emb_method', 'gnn')

    if args.use_emb_coarsen:
        set_seed(args.base_seed)
        t0_pre = time.time()

        if emb_method == 'metapath2vec':
            print("\n" + "="*60)
            print("  MetaPath2Vec EMBEDDING  (unsupervised, topology-based)")
            print("="*60)
            mp2v_dim    = getattr(args, 'mp2v_dim',    128)
            mp2v_epochs = getattr(args, 'mp2v_epochs', 5)
            emb_dict = extract_metapath2vec_embeddings(
                data, embedding_dim=mp2v_dim, epochs=mp2v_epochs, lr=0.01)

        elif emb_method == 'fast' or getattr(args, 'fast_embed', False):
            fe_hops   = getattr(args, 'fast_embed_hops',   2)
            fe_outdim = getattr(args, 'fast_embed_outdim', 128)
            print("\n" + "="*60)
            print(f"  FAST EMBED  (SGC propagation, hops={fe_hops}, out_dim={fe_outdim})")
            print("="*60)
            emb_dict = fast_embed_hetero(
                data, n_hops=fe_hops, out_dim=fe_outdim,
                device=_cfg.DEVICE, verbose=True)

        else:
            _pp = getattr(args, 'pretrain_patience', 0)
            _pretrain_patience_display = _pp if _pp > 0 else max(10, args.patience // 3)
            print("\n" + "="*60)
            print(f"  GNN PRETRAIN  (training labels only -> embeddings)  "
                  f"[patience={_pretrain_patience_display}]")
            if _is_large_graph(data, args):
                nb = getattr(args, 'mini_batch_size', 512)
                print(f"  [Auto] Large graph (>={MINI_BATCH_THRESHOLD} nodes) -> "
                      f"mini-batch mode (batch_size={nb})")
            print("="*60)
            pretrain_hidden = getattr(args, 'pretrain_hidden', None) or args.hidden
            pretrain_model = build_model(data, pretrain_hidden, args.dropout, args.num_layers,
                                         gnn_model=getattr(args, 'gnn_model', 'sage'))
            if pretrain_hidden != args.hidden:
                print(f"  [Pretrain] pretrain_hidden={pretrain_hidden} "
                      f"(training hidden={args.hidden})")
            _pp = getattr(args, 'pretrain_patience', 0)
            _pretrain_patience = _pp if _pp > 0 else max(10, args.patience // 3)
            _orig_patience, args.patience = args.patience, _pretrain_patience
            _train_auto(pretrain_model, data,
                        epochs=args.pretrain_epochs, lr=args.lr,
                        args=args, desc='pretrain')
            args.patience = _orig_patience
            print(f"  [Emb] Extracting embeddings ...")
            emb_dict = _extract_emb_auto(pretrain_model, data, args)
            del pretrain_model
            if _cfg.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        t_pretrain = time.time() - t0_pre
        print(f"  [Embed] Done in {t_pretrain:.1f}s")
        coarsen_features, coarsen_feat_dims = extract_emb_flat_arrays(emb_dict)

    return dict(
        data=data,
        src_nodes=src_nodes, dst_nodes=dst_nodes, weights=weights,
        all_features=all_features, type_boundaries=type_boundaries,
        feature_dims=feature_dims, offsets=offsets,
        emb_dict=emb_dict,
        coarsen_features=coarsen_features,
        coarsen_feat_dims=coarsen_feat_dims,
    )


def _coarsen_from_context(ctx, args):
    """Run coarsening using a pre-loaded context.  Returns (coalition_map, t_coarsen)."""
    print("\n" + "="*60)
    print("  COARSENING")
    print("="*60)

    use_auto = (getattr(args, 'use_auto_coarsen', False)
                and getattr(args, 'coarsen_method', 'hcgc') == 'hcgc'
                and getattr(args, 'target_ratio', 0.0) > 0)

    if use_auto:
        if ctx.get('emb_dict') is None:
            print("  [AutoCoarsen] WARNING: no embeddings in context; "
                  "probe will be random. Consider pretrain=True.")
        cm, t_c, info = auto_coarsen(
            ctx, args,
            target_ratio=args.target_ratio,
            max_acc_loss=getattr(args, 'max_acc_loss', 0.05),
            fast_scale=getattr(args, 'use_fast_scale', False),
        )
    else:
        cm, t_c = _run_coarsen(
            ctx['src_nodes'], ctx['dst_nodes'], ctx['weights'],
            ctx['coarsen_features'], ctx['type_boundaries'],
            ctx['coarsen_feat_dims'], args)
    print(f"\n  Coarsen time: {t_c:.2f}s")

    return cm, t_c


def _train_from_coalition_map(cm, t_coarsen, ctx, args, trial_seed=None):
    """Run eval_pipeline given a pre-computed coalition_map and loaded context."""
    return eval_pipeline(
        'Ours', cm, ctx['data'], ctx['offsets'], ctx['type_boundaries'],
        t_coarsen, args, trial_seed=trial_seed,
        emb_dict=ctx['emb_dict'] if args.use_emb_coarsen else None)
