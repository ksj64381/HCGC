"""
hcgc/_coarsen.py -- Graph coarsening: flat array extraction, HCGC core,
                    auto_coarsen (bracket + binary search), and scale prediction.
"""

import copy
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from hcgc._config import _CFG

try:
    import hcgc_module
    _HCGC_AVAILABLE = True
except ImportError:
    try:
        # Look for pre-built binary in _ext/
        import importlib.util, sys, os, glob as _glob
        _ext_dir = os.path.join(os.path.dirname(__file__), '_ext')
        _pattern = os.path.join(_ext_dir, 'hcgc_module*.pyd') \
            if sys.platform == 'win32' else \
            os.path.join(_ext_dir, 'hcgc_module*.so')
        _candidates = _glob.glob(_pattern)
        if _candidates:
            _spec = importlib.util.spec_from_file_location('hcgc_module', _candidates[0])
            hcgc_module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(hcgc_module)
            sys.modules['hcgc_module'] = hcgc_module
            _HCGC_AVAILABLE = True
        else:
            _HCGC_AVAILABLE = False
    except Exception:
        _HCGC_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Flat array extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_flat_arrays(data):
    """Convert HeteroData to flat numpy arrays expected by the C++ kernel."""
    offsets, feats_list, feat_dims, boundaries = {}, [], [], []
    cur = 0
    for nt in _CFG.node_types:
        n = data[nt].num_nodes
        x = data[nt].x.float()
        norm = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x = (x / norm).numpy().astype(np.float32)
        feats_list.append(x.ravel())
        feat_dims.append(x.shape[1])
        cur += n
        boundaries.append(cur)
        offsets[nt] = cur - n

    all_features    = np.concatenate(feats_list)
    type_boundaries = np.array(boundaries, dtype=np.int32)
    feature_dims    = np.array(feat_dims,  dtype=np.int32)

    src_list, dst_list = [], []
    for et in data.edge_types:
        s_type, _, d_type = et
        if s_type not in offsets or d_type not in offsets:
            continue
        ei = data[et].edge_index.numpy()
        src_list.append(ei[0] + offsets[s_type])
        dst_list.append(ei[1] + offsets[d_type])
        src_list.append(ei[1] + offsets[d_type])
        dst_list.append(ei[0] + offsets[s_type])

    src_nodes = np.concatenate(src_list).astype(np.int32)
    dst_nodes = np.concatenate(dst_list).astype(np.int32)
    weights   = np.ones(len(src_nodes), dtype=np.float32)
    return src_nodes, dst_nodes, weights, all_features, type_boundaries, feature_dims, offsets


# ══════════════════════════════════════════════════════════════════════════════
# Build compressed HeteroData
# ══════════════════════════════════════════════════════════════════════════════

def build_compressed_data(data, coalition_map, offsets, type_boundaries,
                          use_soft_labels=False, emb_dict=None, emb_temp=1.0):
    """Map original nodes to supernodes and return a new HeteroData object.

    Returns: (cdata, local_cm, stats)
      cdata    : compressed HeteroData
      local_cm : dict[node_type -> LongTensor] mapping original -> supernode index
      stats    : dict with node/edge compression counts
    """
    cm = coalition_map.astype(np.int64)

    local_cm, n_super = {}, {}
    for i, nt in enumerate(_CFG.node_types):
        start = offsets[nt]
        end   = int(type_boundaries[i])
        roots_global, inverse = np.unique(cm[start:end], return_inverse=True)
        local_cm[nt] = torch.from_numpy(inverse.astype(np.int64))
        n_super[nt]  = len(roots_global)

    cdata = HeteroData()

    # ── Node features ─────────────────────────────────────────────────────────
    for nt in _CFG.node_types:
        lc  = local_cm[nt]
        ox  = data[nt].x
        D, ns = ox.shape[1], n_super[nt]

        if emb_dict is not None and nt in emb_dict:
            h     = emb_dict[nt].float()
            D_emb = h.shape[1]
            h_ctr = torch.zeros(ns, D_emb)
            h_cnt = torch.zeros(ns, 1)
            h_ctr.scatter_add_(0, lc.unsqueeze(1).expand(-1, D_emb), h)
            h_cnt.scatter_add_(0, lc.unsqueeze(1), torch.ones(len(h), 1))
            h_ctr = h_ctr / h_cnt.clamp(min=1)

            diff    = h - h_ctr[lc]
            dist_sq = (diff * diff).sum(dim=1)

            mem_mean = torch.zeros(ns)
            mem_cnt  = torch.zeros(ns)
            mem_mean.scatter_add_(0, lc, dist_sq)
            mem_cnt.scatter_add_(0, lc, torch.ones(len(h)))
            mem_mean = (mem_mean / mem_cnt.clamp(min=1))[lc]

            safe_d  = (mem_mean * emb_temp).clamp(min=1e-8)
            w       = torch.exp(-dist_sq / safe_d).unsqueeze(1)
            cx      = torch.zeros(ns, D)
            w_sum   = torch.zeros(ns, 1)
            cx.scatter_add_(0, lc.unsqueeze(1).expand(-1, D), ox * w)
            w_sum.scatter_add_(0, lc.unsqueeze(1), w)
            cdata[nt].x = cx / w_sum.clamp(min=1e-8)
        else:
            ox_f = ox.float()
            cx0  = torch.zeros(ns, D)
            cnt0 = torch.zeros(ns, 1)
            cx0.scatter_add_(0, lc.unsqueeze(1).expand(-1, D), ox_f)
            cnt0.scatter_add_(0, lc.unsqueeze(1), torch.ones(len(ox_f), 1))
            centroid = cx0 / cnt0.clamp(min=1)
            diff    = ox_f - centroid[lc]
            dist_sq = (diff * diff).sum(dim=1)
            m_mean  = torch.zeros(ns)
            m_cnt   = torch.zeros(ns)
            m_mean.scatter_add_(0, lc, dist_sq)
            m_cnt.scatter_add_(0, lc, torch.ones(len(ox_f)))
            m_mean = (m_mean / m_cnt.clamp(min=1))[lc]
            w    = torch.exp(-dist_sq / m_mean.clamp(min=1e-8)).unsqueeze(1)
            cx   = torch.zeros(ns, D)
            wsum = torch.zeros(ns, 1)
            cx.scatter_add_(0, lc.unsqueeze(1).expand(-1, D), ox_f * w)
            wsum.scatter_add_(0, lc.unsqueeze(1), w)
            cdata[nt].x = cx / wsum.clamp(min=1e-8)

    # ── Labels + masks ────────────────────────────────────────────────────────
    nt   = _CFG.target_type
    lc_a = local_cm[nt]
    ns_a = n_super[nt]
    y_orig = data[nt].y

    labeled = data[nt].train_mask | data[nt].val_mask | data[nt].test_mask
    one_hot = torch.zeros(len(y_orig), _CFG.num_classes, dtype=torch.long)
    if labeled.any():
        one_hot[labeled] = (
            torch.zeros(int(labeled.sum()), _CFG.num_classes, dtype=torch.long)
            .scatter_(1, y_orig[labeled].unsqueeze(1), 1)
        )
    vote = torch.zeros(ns_a, _CFG.num_classes, dtype=torch.long)
    vote.scatter_add_(0, lc_a.unsqueeze(1).expand(-1, _CFG.num_classes), one_hot)
    cdata[nt].y = vote.argmax(dim=1)

    if use_soft_labels:
        vf = vote.float()
        cdata[nt].soft_y = vf / vf.sum(dim=1, keepdim=True).clamp(min=1)

    purity = vote.max(dim=1).values.float() / vote.sum(dim=1).float().clamp(min=1)
    print(f"  [comp] {nt} supernode purity: {purity.mean():.4f} "
          f"(min {purity.min():.3f}, median {purity.median():.3f})")

    n_tr = torch.zeros(ns_a, dtype=torch.long)
    n_va = torch.zeros(ns_a, dtype=torch.long)
    n_te = torch.zeros(ns_a, dtype=torch.long)
    n_tr.scatter_add_(0, lc_a, data[nt].train_mask.long())
    n_va.scatter_add_(0, lc_a, data[nt].val_mask.long())
    n_te.scatter_add_(0, lc_a, data[nt].test_mask.long())
    has_tr = n_tr > 0
    has_va = n_va > 0
    has_te = n_te > 0
    cdata[nt].train_mask = has_tr
    cdata[nt].val_mask   = has_va & ~has_tr
    cdata[nt].test_mask  = has_te & ~has_tr

    # ── Edges: remap + deduplicate ────────────────────────────────────────────
    orig_edges_total = 0
    comp_edges_total = 0
    edge_stats = {}

    for et in data.edge_types:
        s_type, rel, d_type = et
        if s_type not in local_cm or d_type not in local_cm:
            continue
        ei    = data[et].edge_index
        new_s = local_cm[s_type][ei[0]]
        new_d = local_cm[d_type][ei[1]]
        comp_ei = torch.unique(torch.stack([new_s, new_d]), dim=1)
        cdata[et].edge_index = comp_ei

        n_orig = ei.shape[1]
        n_comp = comp_ei.shape[1]
        orig_edges_total += n_orig
        comp_edges_total += n_comp
        edge_stats[et] = {
            'orig': n_orig,
            'comp': n_comp,
            'ratio': n_orig / max(n_comp, 1),
        }

    if orig_edges_total > 0:
        edge_ratio = orig_edges_total / max(comp_edges_total, 1)
        print(f"  [comp] edges: {orig_edges_total:,} -> {comp_edges_total:,}  "
              f"({(1 - comp_edges_total/orig_edges_total)*100:.1f}% reduced, "
              f"{edge_ratio:.2f}x)")
        for et, v in edge_stats.items():
            print(f"    {str(et):40s}: {v['orig']:7,} -> {v['comp']:6,}  "
                  f"({v['ratio']:.2f}x)")

    stats = {
        'nodes_orig':  sum(int(type_boundaries[i]) - offsets[nt]
                          for i, nt in enumerate(_CFG.node_types)),
        'nodes_comp':  sum(n_super[nt] for nt in _CFG.node_types),
        'edges_orig':  orig_edges_total,
        'edges_comp':  comp_edges_total,
        'edge_ratio':  orig_edges_total / max(comp_edges_total, 1),
        'per_edge_type': edge_stats,
    }

    return cdata, local_cm, stats


# ══════════════════════════════════════════════════════════════════════════════
# PCA feature reduction (for large-graph coarsening speed)
# ══════════════════════════════════════════════════════════════════════════════

def _pca_reduce_features(all_features, feature_dims, type_boundaries, target_dim):
    """Per-type PCA: reduce each node type's features to at most target_dim dims."""
    from sklearn.decomposition import PCA
    result_chunks = []
    new_dims = []
    cursor = 0
    prev_boundary = 0
    for i, dim in enumerate(feature_dims):
        boundary = int(type_boundaries[i])
        n = boundary - prev_boundary
        chunk = all_features[cursor: cursor + n * dim].reshape(n, dim)
        d_out = min(target_dim, dim)
        if d_out < dim and n > d_out:
            pca = PCA(n_components=d_out, random_state=42)
            chunk = pca.fit_transform(chunk).astype(np.float32)
            print(f"  [PCA] type {i}: {dim} -> {d_out} dims "
                  f"(var explained: {pca.explained_variance_ratio_.sum():.3f})")
        else:
            d_out = dim
        result_chunks.append(chunk.reshape(-1))
        new_dims.append(d_out)
        cursor += n * dim
        prev_boundary = boundary
    return np.concatenate(result_chunks), np.array(new_dims, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Core coarsening runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_coarsen(src_nodes, dst_nodes, weights, all_features,
                 type_boundaries, feature_dims, args,
                 mds_override=None, nlvl_override=None):
    # mds_override is unused; retained for API compatibility
    hub_caps = (np.array([], dtype=np.int32) if args.auto_hub_caps or not args.hub_degree_caps.strip()
                else np.array([int(x) for x in args.hub_degree_caps.split(',')], dtype=np.int32))
    nlvl = nlvl_override if nlvl_override is not None else args.num_levels

    coarsen_pca_dim = getattr(args, 'coarsen_pca_dim', 0)
    if coarsen_pca_dim > 0:
        t_pca = time.time()
        all_features, feature_dims = _pca_reduce_features(
            all_features, feature_dims, type_boundaries, coarsen_pca_dim)
        print(f"  [PCA] coarsening features reduced to {coarsen_pca_dim}-dim "
              f"({time.time()-t_pca:.1f}s)")

    t0 = time.time()

    if not _HCGC_AVAILABLE:
        raise RuntimeError(
            "hcgc_module not built.\n"
            "  Option 1 (compile from source):\n"
            "    python setup.py build_ext --inplace\n"
            "  Option 2 (pre-built binary):\n"
            "    Copy a matching .pyd/.so into hcgc/_ext/\n"
            "    See README.md for download links."
        )

    _base_args = (
        src_nodes, dst_nodes, weights,
        all_features, type_boundaries, feature_dims,
        nlvl,
        getattr(args, 'hcgc_inner_passes', 2),
        getattr(args, 'hcgc_max_outer', 10),
        getattr(args, 'hcgc_feat_var_scale', 1.0),
        args.max_candidates, 0, hub_caps,
        args.auto_hub_caps,
    )
    _pt_raw = getattr(args, 'hcgc_feat_var_scale_by_type', None)
    if _pt_raw is not None and len(_pt_raw) > 0:
        _pt_arr = np.array(_pt_raw, dtype=np.float32)
    else:
        _pt_arr = np.array([], dtype=np.float32)

    try:
        cm = hcgc_module.create_graph_hcgc(
            *_base_args,
            getattr(args, 'hcgc_skip_reassignment', False),
            getattr(args, 'hcgc_window_size', 20),
            getattr(args, 'hub_anchor_percentile', 0.0),
            _pt_arr,
            float(getattr(args, 'hcgc_target_comp_ratio', 0.0)),
        )
    except TypeError:
        print("  [warn] Old hcgc_module version - some parameters not supported.\n"
              "         Please rebuild: python setup.py build_ext --inplace")
        cm = hcgc_module.create_graph_hcgc(*_base_args)
    return cm, time.time() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Auto-coarsen support: distance sampling + CDF scale prediction
# ══════════════════════════════════════════════════════════════════════════════

def _sample_normalized_distances(ctx, n_samples=3000, rng_seed=42):
    """Sample edge-pair squared L2 distances in embedding space."""
    rng = np.random.default_rng(rng_seed)

    feats, fdims, boundaries, offsets = (
        ctx['coarsen_features'], ctx['coarsen_feat_dims'],
        ctx['type_boundaries'],  ctx['offsets'])

    feat_by_type = {}
    cursor = 0
    for i, nt in enumerate(_CFG.node_types):
        s = offsets[nt]; e = int(boundaries[i])
        n = e - s; d = int(fdims[i])
        feat_by_type[nt] = (s, feats[cursor: cursor + n * d].reshape(n, d).astype(np.float64))
        cursor += n * d

    sigma2 = 1.0

    src_arr = np.asarray(ctx['src_nodes'], dtype=np.int64)
    dst_arr = np.asarray(ctx['dst_nodes'], dtype=np.int64)

    n_total = int(boundaries[-1])
    type_of = np.full(n_total, -1, dtype=np.int32)
    for i, nt in enumerate(_CFG.node_types):
        type_of[offsets[nt]: int(boundaries[i])] = i
    same_mask = (type_of[src_arr] == type_of[dst_arr]) & (type_of[src_arr] >= 0)
    same_idx  = np.where(same_mask)[0]

    d_norms = []
    if len(same_idx) >= 10:
        pick = rng.choice(same_idx, min(n_samples, len(same_idx)), replace=False)
        for k in pick:
            u, v = int(src_arr[k]), int(dst_arr[k])
            nt   = _CFG.node_types[int(type_of[u])]
            s, x = feat_by_type[nt]
            diff = x[u - s] - x[v - s]
            d_norms.append(float((diff * diff).sum()))
    else:
        nt_t   = _CFG.target_type
        s_t, x_t = feat_by_type[nt_t]
        n_t = len(x_t)
        pick_u = rng.choice(n_t, min(n_samples, n_t), replace=False)
        pick_v = rng.choice(n_t, min(n_samples, n_t), replace=False)
        diff = x_t[pick_u] - x_t[pick_v]
        d_norms = list((diff * diff).sum(axis=1))

    d_norms = np.array(d_norms, dtype=np.float64)
    d_norms = d_norms[d_norms > 0]
    return np.sort(d_norms), sigma2


def predict_scale_for_compression(ctx, target_ratio,
                                  n_samples=3000, verbose=True):
    """CDF-only estimate of feat_var_scale for a given target node retention ratio.

    Args:
        target_ratio: fraction of nodes to keep, e.g. 0.1 = keep 10% = 10x compression.
    """
    _tc = 1.0 / max(float(target_ratio), 1e-6)
    d_norms, _ = _sample_normalized_distances(ctx, n_samples)
    if len(d_norms) == 0:
        if verbose:
            print("  [AutoCoarsen] Warning: no edge distance samples; using scale=0.4")
        return 0.4, []

    log_c = math.log(max(_tc, 1.01))
    pct = float(np.clip(0.2 + (log_c / 5.0) * 0.65, 0.05, 0.95))
    predicted = float(np.percentile(d_norms, pct * 100))
    if verbose:
        print(f"  [AutoCoarsen] CDF-only estimate: scale={predicted:.4f} "
              f"(p{pct*100:.0f} of edge-pair distances -> {_tc:.1f}x)")
    return predicted, []


# ══════════════════════════════════════════════════════════════════════════════
# One-shot scale prediction (no HCGC runs required)
# ══════════════════════════════════════════════════════════════════════════════

def _sample_mediator_pair_energies(ctx, n_samples=3000, rng_seed=42):
    """Sample normalised Dirichlet energies of same-type mediator-path pairs."""
    from collections import defaultdict

    rng = np.random.default_rng(rng_seed)

    src_arr  = np.asarray(ctx['src_nodes'], dtype=np.int64)
    dst_arr  = np.asarray(ctx['dst_nodes'], dtype=np.int64)
    w_arr    = np.asarray(ctx['weights'],   dtype=np.float32)
    tb       = ctx['type_boundaries']
    offsets  = ctx['offsets']
    data     = ctx['data']
    emb_dict = ctx.get('emb_dict') or {}

    node_types = _CFG.node_types
    n_total    = int(tb[-1])

    type_of = np.full(n_total, -1, dtype=np.int32)
    for i, nt in enumerate(node_types):
        type_of[offsets[nt]: int(tb[i])] = i

    feat_by_type     = {}
    feat_var_by_type = {}
    for nt in node_types:
        s = offsets[nt]
        if nt in emb_dict:
            x = emb_dict[nt]
            if hasattr(x, 'detach'):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
        elif hasattr(data[nt], 'x') and data[nt].x is not None:
            x = data[nt].x.float().numpy()
        else:
            continue
        feat_by_type[nt]     = (s, x)
        feat_var_by_type[nt] = float(np.mean(np.var(x, axis=0).clip(min=1e-12)))

    if not feat_by_type:
        return np.array([]), feat_var_by_type, {}

    n_sample_med = min(n_samples * 4, n_total)
    sampled_meds = rng.choice(n_total, n_sample_med, replace=False)
    sampled_set  = set(sampled_meds.tolist())

    adj = defaultdict(list)
    for u, v, w in zip(src_arr, dst_arr, w_arr):
        u, v = int(u), int(v)
        if u in sampled_set:
            adj[u].append((v, float(w)))
        if v in sampled_set:
            adj[v].append((u, float(w)))

    _MAX_NBRS       = 20
    _MAX_PAIRS_MED  = 5

    energies              = []
    med_degs_by_type      = defaultdict(list)

    for m in sampled_meds:
        if len(energies) >= n_samples:
            break

        nbrs_by_type = defaultdict(list)
        for v, w in adj.get(m, []):
            t = int(type_of[v])
            if t >= 0:
                nbrs_by_type[t].append((v, w))

        for t_src, nbrs in nbrs_by_type.items():
            if len(nbrs) < 2:
                continue
            nt = node_types[t_src]
            if nt not in feat_by_type:
                continue

            s_t, x_t = feat_by_type[nt]
            fv        = feat_var_by_type.get(nt, 1.0)

            med_degs_by_type[t_src].append(len(nbrs))

            if len(nbrs) > _MAX_NBRS:
                idx  = rng.choice(len(nbrs), _MAX_NBRS, replace=False)
                nbrs = [nbrs[i] for i in idx]

            pairs_done = 0
            for i in range(len(nbrs)):
                if pairs_done >= _MAX_PAIRS_MED:
                    break
                u, w_um = nbrs[i]
                for j in range(i + 1, len(nbrs)):
                    if pairs_done >= _MAX_PAIRS_MED:
                        break
                    v, w_mv = nbrs[j]
                    diff  = (x_t[u - s_t].astype(np.float64)
                             - x_t[v - s_t].astype(np.float64))
                    d2    = float(np.dot(diff, diff))
                    energy = float(w_um * w_mv) * d2 / fv
                    if np.isfinite(energy) and energy > 0:
                        energies.append(energy)
                    pairs_done += 1

    energies = np.sort(np.asarray(energies, dtype=np.float64))
    avg_med_deg_by_type = {
        t: float(np.mean(degs)) for t, degs in med_degs_by_type.items() if degs
    }
    return energies, feat_var_by_type, avg_med_deg_by_type


def predict_scale_one_shot(ctx, target_ratio, n_samples=3000, verbose=True):
    """Predict feat_var_scale for a target node retention ratio WITHOUT any HCGC runs.

    Args:
        target_ratio: fraction of nodes to keep, e.g. 0.1 = keep 10% = 10x compression.
    """
    t0 = time.time()

    energies, feat_var_by_type, avg_med_deg_by_type = \
        _sample_mediator_pair_energies(ctx, n_samples)

    _tc = 1.0 / max(float(target_ratio), 1e-6)

    if len(energies) < 20:
        if verbose:
            print("  [OneShot] Too few mediator pairs; falling back to CDF-edge estimate")
        scale, _ = predict_scale_for_compression(ctx, target_ratio, verbose=verbose)
        return scale, {'fallback': True, 'n_samples': len(energies)}

    if avg_med_deg_by_type:
        tt = _CFG.target_type
        t_tgt = next(
            (i for i, nt in enumerate(_CFG.node_types) if nt == tt), None)
        if t_tgt is not None and t_tgt in avg_med_deg_by_type:
            avg_med_deg = avg_med_deg_by_type[t_tgt]
        else:
            avg_med_deg = float(np.median(list(avg_med_deg_by_type.values())))
    else:
        avg_med_deg = 3.0

    alpha = float(np.clip(avg_med_deg / 2.0, 1.0, 6.0))
    p_target = (1.0 - 1.0 / max(_tc, 1.01)) / alpha
    p_target = float(np.clip(p_target, 0.005, 0.90))
    predicted_scale = float(np.percentile(energies, p_target * 100))
    predicted_scale = max(predicted_scale, 1e-6)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [OneShot] {len(energies)} mediator pairs sampled  "
              f"avg_med_deg={avg_med_deg:.1f}  alpha={alpha:.2f}  "
              f"p_target={p_target:.3f}")
        print(f"  [OneShot] predicted_scale={predicted_scale:.5f}  ({elapsed:.2f}s)")

    info = {
        'n_samples':          len(energies),
        'avg_med_deg':        avg_med_deg,
        'alpha':              alpha,
        'p_target':           p_target,
        'predicted_scale':    predicted_scale,
        'elapsed_sampling':   round(elapsed, 3),
        'fallback':           False,
    }
    return predicted_scale, info


# ══════════════════════════════════════════════════════════════════════════════
# Probe helpers
# ══════════════════════════════════════════════════════════════════════════════

def _probe_from_coalition_map(coalition_map, ctx):
    """Compute linear probe accuracy from a coalition map (no GNN training)."""
    from sklearn.linear_model import LogisticRegression

    tt = _CFG.target_type
    if tt is None:
        return float('nan')
    data     = ctx['data']
    offsets  = ctx['offsets']
    emb_dict = ctx.get('emb_dict') or {}

    if tt not in emb_dict:
        return float('nan')
    emb = emb_dict[tt]
    if hasattr(emb, 'detach'):
        emb = emb.detach().cpu().numpy()
    emb = np.array(emb, dtype=np.float32)

    y = data[tt].y.cpu().numpy()
    s = offsets[tt]; n = len(y)
    cm_t = coalition_map[s: s + n]

    sn_ids, inverse = np.unique(cm_t, return_inverse=True)
    sn_emb = np.zeros((len(sn_ids), emb.shape[1]), dtype=np.float32)
    np.add.at(sn_emb, inverse, emb)
    sn_cnt = np.bincount(inverse).reshape(-1, 1)
    sn_emb /= sn_cnt.clip(min=1)

    X = sn_emb[inverse]
    Y = y
    valid = Y >= 0
    X, Y = X[valid], Y[valid]

    if len(np.unique(Y)) < 2 or len(X) < 20:
        return float('nan')

    try:
        tr_mask = data[tt].train_mask.numpy()[valid]
        X_tr, Y_tr = X[tr_mask], Y[tr_mask]
        X_va, Y_va = X[~tr_mask], Y[~tr_mask]
        if len(X_tr) < 5 or len(X_va) < 5:
            raise ValueError("too few samples")
    except Exception:
        rng2 = np.random.default_rng(42)
        idx  = rng2.permutation(len(X))
        sp   = max(1, int(len(X) * 0.8))
        X_tr, Y_tr = X[idx[:sp]], Y[idx[:sp]]
        X_va, Y_va = X[idx[sp:]], Y[idx[sp:]]

    try:
        clf = LogisticRegression(max_iter=500, random_state=42, n_jobs=1)
        clf.fit(X_tr, Y_tr)
        return float(clf.score(X_va, Y_va))
    except Exception:
        return float('nan')


def _baseline_probe(ctx):
    """Probe accuracy with identity coalition map (no compression)."""
    n_total = int(ctx['type_boundaries'][-1])
    identity_cm = np.arange(n_total, dtype=np.int32)
    return _probe_from_coalition_map(identity_cm, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# Cheap coarsening run helper
# ══════════════════════════════════════════════════════════════════════════════

def _run_coarsen_cheap(ctx, args, scale):
    """Run HCGC coarsening at the given feat_var_scale; return (cm, comp, t)."""
    args.hcgc_feat_var_scale = scale
    n_total = int(ctx['type_boundaries'][-1])
    cm, t = _run_coarsen(
        ctx['src_nodes'], ctx['dst_nodes'], ctx['weights'],
        ctx['coarsen_features'], ctx['type_boundaries'],
        ctx['coarsen_feat_dims'], args)
    comp = n_total / max(len(np.unique(cm)), 1)
    return cm, comp, t


# ══════════════════════════════════════════════════════════════════════════════
# Auto-coarsen: bracket + binary search
# ══════════════════════════════════════════════════════════════════════════════

def auto_coarsen(ctx, args, target_ratio=0.2, max_acc_loss=0.05,
                 max_search_runs=8, verbose=True, fast_scale=False):
    """Find the optimal HCGC scale in a small number of coarsening runs.

    Args:
        target_ratio: fraction of nodes to keep, e.g. 0.1 = keep 10% = 10x compression.

    Algorithm  (default: bracket + binary search)
    ---------
    Phase 1 - Bracket search  (up to ~4 coarsening runs):
      Start at the CDF estimate; repeatedly halve/double until a bracket
      [lo_scale, hi_scale] is found where lo_scale gives comp < target and
      hi_scale gives comp >= target.

    Phase 2 - Geometric binary search  (up to max_search_runs more runs):
      Bisect in log-scale space until compression is within 15% of target.

    Fast-scale mode  (fast_scale=True)
    -----------------------------------
    Uses mediator-pair sampling to predict the scale WITHOUT any HCGC runs,
    then verifies with 1 HCGC run and applies iterative power-law corrections.
    Typically 2-3 total runs vs 5-9 for bracket+binary.
    """
    target_compression = 1.0 / max(float(target_ratio), 1e-6)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  AUTO-COARSEN  target={target_ratio:.3f} ({target_compression:.1f}x)")
        print("=" * 60)

    calib_args = copy.deepcopy(args)

    _tbounds = ctx['type_boundaries']

    def _overall_comp(cm):
        return int(_tbounds[-1]) / max(len(np.unique(cm)), 1)

    _pertype_base = list(getattr(args, 'hcgc_feat_var_scale_by_type', None) or [])
    _use_pertype  = bool(_pertype_base)

    t0 = time.time()
    baseline = _baseline_probe(ctx)
    if verbose:
        print(f"  [AutoCoarsen] Baseline probe: {baseline:.4f}  ({time.time()-t0:.1f}s)")

    cdf_scale, _ = predict_scale_for_compression(
        ctx, target_ratio, verbose=False)

    run_log     = []
    seen_scales = set()

    def _probe_loss(cm):
        p = _probe_from_coalition_map(cm, ctx)
        return baseline - p, p

    def _record(scale, cm, comp, t):
        # Skip per-run probe during search: LogisticRegression on the full
        # dataset (e.g. 26k nodes, 256-dim) costs ~2s per call.  We compute
        # probe accuracy only once for the final best result (see below).
        run_log.append({'scale': scale, 'comp': comp,
                        'probe': float('nan'), 'probe_loss': float('nan'),
                        'cm': cm, 't': t})
        if verbose:
            print(f"    scale={scale:.5f}  comp={comp:.2f}x")
        return float('nan')

    _max_levels = getattr(args, 'hcgc_num_levels', 5)
    calib_args.num_levels             = _max_levels
    calib_args.hcgc_target_comp_ratio = float(target_compression)
    if verbose:
        print(f"  [AutoCoarsen] Multi-level: up to {_max_levels} levels, "
              f"early-stop at {target_compression:.1f}x")

    def _run_if_new(scale):
        key = round(scale, 6)
        if key in seen_scales:
            existing = next((r for r in run_log if round(r['scale'], 6) == key), None)
            if existing:
                return existing['cm'], existing['comp'], existing['t']
        seen_scales.add(key)

        if _use_pertype:
            calib_args.hcgc_feat_var_scale_by_type = [v * scale for v in _pertype_base]
            cm_raw, _, t = _run_coarsen_cheap(
                ctx, calib_args, calib_args.hcgc_feat_var_scale)
        else:
            cm_raw, _, t = _run_coarsen_cheap(ctx, calib_args, scale)

        return cm_raw, _overall_comp(cm_raw), t

    def _writeback(s):
        if _use_pertype:
            args.hcgc_feat_var_scale_by_type = [v * s for v in _pertype_base]
        else:
            args.hcgc_feat_var_scale = s

    def _make_info(comp, probe_after, probe_loss_, scale_, n_runs, saturated=False):
        return {
            'compression':    round(comp, 4),
            'probe_baseline': round(baseline, 4),
            'probe_after':    round(probe_after, 4),
            'probe_loss':     round(probe_loss_, 4),
            'scale_used':     round(float(scale_), 6),
            'n_coarsen_runs': n_runs,
            'saturated':      saturated,
            'all_runs':       [{'scale': r['scale'], 'comp': r['comp'],
                                'probe_loss': r['probe_loss']} for r in run_log],
        }

    # ── Fast-scale mode ───────────────────────────────────────────────────────
    if fast_scale:
        if verbose:
            print(f"\n  [AutoCoarsen] Fast-scale mode  (one-shot + up to 3 corrections)")

        one_shot_scale, one_shot_info = predict_scale_one_shot(
            ctx, target_ratio, verbose=verbose)
        cm_os, comp_os, t_os = _run_if_new(one_shot_scale)
        _record(one_shot_scale, cm_os, comp_os, t_os)

        rel_err = abs(comp_os - target_compression) / max(target_compression, 1.0)
        if verbose:
            print(f"  [AutoCoarsen] One-shot: scale={one_shot_scale:.5f}  "
                  f"comp={comp_os:.2f}x  err={rel_err*100:.1f}%")

        if rel_err <= 0.20:
            best_cm, best_comp, best_t, mid_scale = cm_os, comp_os, t_os, one_shot_scale
        else:
            _MAX_CORR = 3
            _BETA_OVER = 2.5
            _BETA_UNDER = 0.8
            _TOL = 0.25

            cur_scale = one_shot_scale
            cur_comp  = comp_os
            best_cm, best_comp, best_t, mid_scale = cm_os, comp_os, t_os, one_shot_scale

            for _corr_i in range(_MAX_CORR):
                cur_err = abs(cur_comp - target_compression) / max(target_compression, 1.0)
                if cur_err <= _TOL:
                    break

                _BETA = _BETA_OVER if cur_comp > target_compression else _BETA_UNDER
                ratio = target_compression / max(cur_comp, 1e-3)
                new_scale = cur_scale * (ratio ** _BETA)
                new_scale = max(new_scale, 1e-6)

                if verbose:
                    direction = "reduce" if cur_comp > target_compression else "increase"
                    print(f"  [AutoCoarsen] Correction {_corr_i+1}/{_MAX_CORR} "
                          f"({direction}): scale {cur_scale:.5f} -> {new_scale:.5f}  "
                          f"(ratio={ratio:.3f}, beta={_BETA})")

                cm_n, comp_n, t_n = _run_if_new(new_scale)
                _record(new_scale, cm_n, comp_n, t_n)

                new_err = abs(comp_n - target_compression) / max(target_compression, 1.0)
                if new_err < abs(best_comp - target_compression) / max(target_compression, 1.0):
                    best_cm, best_comp, best_t, mid_scale = cm_n, comp_n, t_n, new_scale

                cur_scale, cur_comp = new_scale, comp_n

        _writeback(mid_scale)
        final_probe_loss, final_probe = _probe_loss(best_cm)
        n_runs = len(run_log)

        if verbose:
            print(f"\n  [AutoCoarsen] FINAL  ({n_runs} coarsening runs, fast-scale)")
            print(f"    compression = {best_comp:.2f}x  (target {target_compression:.1f}x)")
            if _use_pertype:
                eff = [round(v * mid_scale, 4) for v in _pertype_base]
                print(f"    multiplier  = {mid_scale:.5f}  (per-type: {eff})")
            else:
                print(f"    scale used  = {mid_scale:.5f}")
            print(f"    probe_acc   = {final_probe:.4f}  "
                  f"(baseline {baseline:.4f},  loss {final_probe_loss:+.4f})")
            if not math.isnan(final_probe_loss) and final_probe_loss > max_acc_loss:
                print(f"  [AutoCoarsen] WARNING: probe_loss={final_probe_loss:.4f} "
                      f"> {max_acc_loss:.3f} -- significant accuracy drop expected.")

        return best_cm, best_t, _make_info(
            best_comp, final_probe, final_probe_loss, mid_scale, n_runs)

    # ── Phase 1: bracket search ───────────────────────────────────────────────
    if verbose:
        print(f"\n  [AutoCoarsen] Phase 1: bracket search  (CDF seed={cdf_scale:.4f})")

    hi_scale = cdf_scale
    cm_h, comp_h, t_h = _run_if_new(hi_scale)
    _record(hi_scale, cm_h, comp_h, t_h)

    if comp_h < target_compression:
        # CDF scale too small: not enough compression.
        # Use power-law jump (ratio-based) instead of fixed ×2.
        lo_scale = hi_scale
        lo_comp  = comp_h
        prev_comp = comp_h
        plateau_count = 0
        for _ in range(6):
            # Jump by the compression ratio (power-law alpha≈1 estimate),
            # clamped to [2×, 16×] to avoid overshoot on non-linear curves.
            ratio    = target_compression / max(lo_comp, 1.0)
            hi_scale = lo_scale * min(max(ratio, 2.0), 16.0)
            cm_h, comp_h, t_h = _run_if_new(hi_scale)
            _record(hi_scale, cm_h, comp_h, t_h)
            if comp_h >= target_compression:
                break
            if comp_h < prev_comp * 1.10:
                plateau_count += 1
                if plateau_count >= 2:
                    if verbose:
                        print(f"  [AutoCoarsen] WARNING: compression saturated "
                              f"at ~{comp_h:.1f}x < target {target_compression:.1f}x. "
                              f"Returning best achievable.")
                    best_entry = max(run_log, key=lambda r: r['comp'])
                    pl2, pr2   = _probe_loss(best_entry['cm'])
                    mid_scale  = best_entry['scale']
                    _writeback(mid_scale)
                    return best_entry['cm'], best_entry['t'], _make_info(
                        best_entry['comp'], pr2, pl2, mid_scale,
                        len(run_log), saturated=True)
            else:
                plateau_count = 0
            prev_comp = comp_h
            lo_scale  = hi_scale
            lo_comp   = comp_h
        lo_scale_val = lo_scale
    else:
        # CDF scale too large: too much compression.
        # Use power-law jump (divide by ratio) instead of fixed /2.
        lo_scale = hi_scale
        cur_comp = comp_h
        for _ in range(5):
            # Jump down by the compression ratio, clamped to [2×, 16×].
            ratio    = cur_comp / max(target_compression, 1.0)
            lo_scale = lo_scale / min(max(ratio, 2.0), 16.0)
            cm_l, comp_l, t_l = _run_if_new(lo_scale)
            _record(lo_scale, cm_l, comp_l, t_l)
            if comp_l < target_compression:
                break
            cur_comp = comp_l
        lo_scale_val = lo_scale
        hi_scale     = cdf_scale

    if verbose:
        print(f"  [AutoCoarsen] Bracket: [{lo_scale_val:.5f}, {hi_scale:.5f}]")

    # ── Phase 2: log-linear inverse interpolation (regula falsi in log-space) ──
    # Assumes comp ∝ scale^alpha (power law).  Given bracket [lo_s, hi_s] with
    # known compressions [lo_comp, hi_comp], solve directly for the target scale
    # instead of bisecting.  Converges in 2-3 iterations vs 8 for bisection.
    if verbose:
        print(f"\n  [AutoCoarsen] Phase 2: log-linear interpolation "
              f"(max {max_search_runs} runs)")

    lo_s, hi_s = lo_scale_val, hi_scale
    best_cm, best_comp, best_t, mid_scale = None, None, None, hi_scale

    # Retrieve bracket endpoint compressions from run_log.
    def _lookup_comp(scale):
        tol = max(abs(scale) * 0.001, 1e-9)
        matches = [r for r in run_log if abs(r['scale'] - scale) < tol]
        return matches[-1]['comp'] if matches else None

    lo_comp = _lookup_comp(lo_s)
    hi_comp = _lookup_comp(hi_s)

    for _ in range(max_search_runs):
        # --- Log-linear inverse interpolation ---
        # If we have valid bracket compressions, predict the target scale directly.
        mid_scale = None
        if (lo_comp is not None and hi_comp is not None
                and lo_comp > 0 and hi_comp > lo_comp
                and lo_comp < target_compression <= hi_comp
                and lo_s < hi_s):
            try:
                alpha = ((math.log(hi_comp) - math.log(lo_comp)) /
                         (math.log(max(hi_s, 1e-9)) - math.log(max(lo_s, 1e-9))))
                if alpha > 0.05:
                    log_s_tgt = (math.log(max(lo_s, 1e-9)) +
                                 (math.log(target_compression) - math.log(lo_comp)) / alpha)
                    mid_scale = math.exp(log_s_tgt)
                    # Clamp strictly inside bracket to avoid degenerate repeats.
                    mid_scale = max(lo_s * 1.0001, min(hi_s * 0.9999, mid_scale))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        if mid_scale is None:   # Fallback: geometric bisection
            mid_scale = math.exp(
                (math.log(max(lo_s, 1e-8)) + math.log(max(hi_s, 1e-8))) / 2)

        cm_m, comp_m, t_m = _run_if_new(mid_scale)
        _record(mid_scale, cm_m, comp_m, t_m)

        rel_err = abs(comp_m - target_compression) / max(target_compression, 1)
        if rel_err < 0.15:
            best_cm, best_comp, best_t = cm_m, comp_m, t_m
            if verbose:
                print(f"  [AutoCoarsen] Close enough ({rel_err*100:.1f}% error). "
                      f"Stopping search.")
            break

        if comp_m < target_compression:
            lo_s, lo_comp = mid_scale, comp_m
        else:
            hi_s, hi_comp = mid_scale, comp_m
            best_cm, best_comp, best_t = cm_m, comp_m, t_m

    if best_cm is None:
        best_entry = min(run_log, key=lambda r: abs(r['comp'] - target_compression))
        best_cm, best_comp, best_t = (
            best_entry['cm'], best_entry['comp'], best_entry['t'])
        mid_scale = best_entry['scale']

    _writeback(mid_scale)
    final_probe_loss, final_probe = _probe_loss(best_cm)
    n_runs = len(run_log)

    if verbose:
        print(f"\n  [AutoCoarsen] FINAL  ({n_runs} coarsening runs)")
        print(f"    compression = {best_comp:.2f}x  (target {target_compression:.1f}x)")
        if _use_pertype:
            eff = [round(v * mid_scale, 4) for v in _pertype_base]
            print(f"    multiplier  = {mid_scale:.5f}  (per-type: {eff})")
        else:
            print(f"    scale used  = {mid_scale:.5f}")
        print(f"    probe_acc   = {final_probe:.4f}  "
              f"(baseline {baseline:.4f},  loss {final_probe_loss:+.4f})")
        if not math.isnan(final_probe_loss) and final_probe_loss > max_acc_loss:
            print(f"  [AutoCoarsen] WARNING: probe_loss={final_probe_loss:.4f} "
                  f"> {max_acc_loss:.3f} -- significant accuracy drop expected.")

    return best_cm, best_t, _make_info(
        best_comp, final_probe, final_probe_loss, mid_scale, n_runs)
