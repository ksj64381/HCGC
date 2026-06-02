"""
hcgc/_baselines.py -- Lightweight baseline coarseners for apples-to-apples
benchmarking with the existing HCGC evaluation pipeline.
"""

import math
import time
import numpy as np
import torch

from hcgc._api import (
    HCGCResult,
    _apply_freeze_node_types,
    _detect_target_type,
    _ensure_node_features,
    _target_embedding_diagnostics,
)
from hcgc._coarsen import build_compressed_data, extract_flat_arrays
from hcgc._config import _CFG, set_seed


def compress_random_type(data, ratio=0.1, target_type=None,
                         edge_weight_mode='binary', use_soft_labels=False,
                         freeze_node_types=None, seed=42, verbose=True):
    """Random type-isolated coarsening.

    This is a lower-bound sanity baseline: each node type is compressed to the
    requested retention ratio by randomly partitioning nodes into type-local
    buckets, then the normal quotient-graph builder is reused.
    """
    return _compress_by_order(
        data,
        ratio=ratio,
        target_type=target_type,
        order_fn=_random_order,
        name='random_type',
        edge_weight_mode=edge_weight_mode,
        use_soft_labels=use_soft_labels,
        freeze_node_types=freeze_node_types,
        seed=seed,
        verbose=verbose,
    )


def compress_ahugc_style(data, ratio=0.1, target_type=None,
                         edge_weight_mode='binary', use_soft_labels=False,
                         freeze_node_types=None, hash_bits=16, seed=42,
                         verbose=True):
    """AH-UGC-style hash/clockwise type-isolated coarsening.

    The implementation is intentionally labelled "style": it uses normalized
    raw features plus a log-degree signal, random-projection LSH signatures, and
    adjacent buckets in hash order. It is useful as a fast comparison point, but
    should not be described as official AH-UGC code.
    """
    return _compress_by_order(
        data,
        ratio=ratio,
        target_type=target_type,
        order_fn=lambda data, offsets, tb, rng: _lsh_order(
            data, offsets, tb, rng, hash_bits=hash_bits),
        name='ahugc_style',
        edge_weight_mode=edge_weight_mode,
        use_soft_labels=use_soft_labels,
        freeze_node_types=freeze_node_types,
        seed=seed,
        verbose=verbose,
    )


def _compress_by_order(data, ratio, target_type, order_fn, name,
                       edge_weight_mode='binary', use_soft_labels=False,
                       freeze_node_types=None, seed=42, verbose=True):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    t_total = time.perf_counter()

    data = _ensure_node_features(data)
    _CFG.node_types = list(data.node_types)
    _CFG.target_type = _detect_target_type(data, target_type)
    _CFG.num_classes = int(data[_CFG.target_type].y.max().item()) + 1
    _CFG.dataset = None

    _, _, _, _, type_boundaries, _, offsets = extract_flat_arrays(
        data, l2_normalize=False)
    n_total = int(type_boundaries[-1])

    t_order = time.perf_counter()
    orders, rep_dict = order_fn(data, offsets, type_boundaries, rng)
    cm = _partition_by_type_orders(data, offsets, type_boundaries, orders, ratio)
    t_coarsen = time.perf_counter() - t_order

    cm = _apply_freeze_node_types(
        cm, offsets, type_boundaries, _CFG.node_types,
        freeze_node_types, verbose)

    t_build = time.perf_counter()
    cdata, local_cm, stats = build_compressed_data(
        data, cm, offsets, type_boundaries,
        use_soft_labels=use_soft_labels,
        emb_dict=rep_dict,
        edge_weight_mode=edge_weight_mode,
    )
    emb_diag = _target_embedding_diagnostics(
        rep_dict, local_cm, _CFG.target_type)
    t_build = time.perf_counter() - t_build

    n_comp = int(stats['nodes_comp'])
    actual_ratio = n_comp / max(n_total, 1)
    info = {
        'compressor': name,
        'compression': round(1.0 / max(actual_ratio, 1e-12), 4),
        'n_nodes_orig': n_total,
        'n_nodes_comp': n_comp,
        'coarsen_time': round(t_coarsen, 2),
        'build_time': round(t_build, 2),
        'nodes_orig': stats['nodes_orig'],
        'nodes_comp': stats['nodes_comp'],
        'edges_orig': stats['edges_orig'],
        'edges_comp': stats['edges_comp'],
        'edge_ratio': round(stats['edge_ratio'], 4),
        'freeze_node_types': list(freeze_node_types or []),
        'target_emb_distortion': emb_diag['distortion'],
        'target_emb_cosine': emb_diag['cosine'],
    }

    if verbose:
        print(f"[{name}] Done: {n_total:,} -> {n_comp:,} nodes  "
              f"(actual ratio={actual_ratio:.3f}, "
              f"{1.0 / max(actual_ratio, 1e-12):.1f}x)  "
              f"total={time.perf_counter() - t_total:.2f}s")

    return HCGCResult(
        data=cdata,
        ratio=actual_ratio,
        node_map=local_cm,
        info=info,
    )


def _partition_by_type_orders(data, offsets, type_boundaries, orders, ratio):
    n_total = int(type_boundaries[-1])
    cm = np.empty(n_total, dtype=np.int64)

    for i, nt in enumerate(_CFG.node_types):
        start = int(offsets[nt])
        end = int(type_boundaries[i])
        n = end - start
        if n <= 0:
            continue

        order = np.asarray(orders[nt], dtype=np.int64)
        if len(order) != n:
            raise ValueError(f"order for node type {nt!r} has len={len(order)}, "
                             f"expected {n}")

        k = max(1, int(math.ceil(float(ratio) * n)))
        splits = np.array_split(order, k)
        for bucket in splits:
            if len(bucket) == 0:
                continue
            root = start + int(bucket[0])
            cm[start + bucket] = root
    return cm


def _random_order(data, offsets, type_boundaries, rng):
    orders = {}
    rep_dict = {}
    for i, nt in enumerate(_CFG.node_types):
        n = int(type_boundaries[i]) - int(offsets[nt])
        orders[nt] = rng.permutation(n)
        rep_dict[nt] = _node_representation(data, nt)
    return orders, rep_dict


def _lsh_order(data, offsets, type_boundaries, rng, hash_bits=16):
    orders = {}
    rep_dict = {}
    hash_bits = max(1, min(int(hash_bits), 30))

    for nt in _CFG.node_types:
        rep = _node_representation(data, nt)
        rep_dict[nt] = rep
        n, dim = rep.shape
        if n == 0:
            orders[nt] = np.empty(0, dtype=np.int64)
            continue

        w = torch.from_numpy(
            rng.standard_normal((dim, hash_bits)).astype(np.float32))
        proj = rep @ w
        bits = (proj > 0).to(torch.int64).cpu().numpy()
        powers = (1 << np.arange(hash_bits, dtype=np.int64))
        hash_val = bits @ powers

        # Secondary scalar projection gives a stable clockwise order inside
        # identical hash buckets without introducing label information.
        score_w = torch.from_numpy(
            rng.standard_normal((dim, 1)).astype(np.float32))
        score = (rep @ score_w).squeeze(1).cpu().numpy()
        orders[nt] = np.lexsort((score, hash_val)).astype(np.int64)
    return orders, rep_dict


def _node_representation(data, nt):
    x = data[nt].x.float().detach().cpu()
    x = _row_l2_normalize(x)
    deg = _degree_feature(data, nt)
    return torch.cat([x, deg], dim=1).contiguous()


def _row_l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _degree_feature(data, nt):
    n = data[nt].num_nodes
    deg = torch.zeros(n, dtype=torch.float)
    for et in data.edge_types:
        s_type, _, d_type = et
        ei = data[et].edge_index
        if s_type == nt:
            deg.scatter_add_(0, ei[0].cpu(), torch.ones(ei.shape[1]))
        if d_type == nt:
            deg.scatter_add_(0, ei[1].cpu(), torch.ones(ei.shape[1]))
    deg = torch.log1p(deg).unsqueeze(1)
    return _row_l2_normalize(deg)
