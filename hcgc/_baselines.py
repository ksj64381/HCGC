"""
hcgc/_baselines.py -- Lightweight baseline coarseners for apples-to-apples
benchmarking with the existing HCGC evaluation pipeline.
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

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


def compress_cgc_homo(data, ratio=0.1, target_type=None,
                      edge_weight_mode='binary', use_soft_labels=False,
                      freeze_node_types=None, max_hub_degree=512,
                      max_candidates=128, seed=42, verbose=True):
    """Naive CGC-style homogeneous adaptation for heterogeneous graphs.

    The graph is collapsed to a single untyped adjacency for leader ordering
    and 2-hop neighbour exploration.  To keep the existing heterogeneous
    downstream pipeline valid, accepted merges remain type-constrained: a node
    can only join a coalition of the same node type.  This baseline is intended
    as "what if we apply CGC's homogeneous local exploration to a hetero graph?"
    rather than official CGC code.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    del rng  # deterministic implementation; keep seed call for consistency
    t_total = time.perf_counter()

    data = _ensure_node_features(data)
    _CFG.node_types = list(data.node_types)
    _CFG.target_type = _detect_target_type(data, target_type)
    _CFG.num_classes = int(data[_CFG.target_type].y.max().item()) + 1
    _CFG.dataset = None

    _, _, _, _, type_boundaries, _, offsets = extract_flat_arrays(
        data, l2_normalize=False)
    n_total = int(type_boundaries[-1])
    rep_dict = {nt: _node_representation(data, nt) for nt in _CFG.node_types}

    t_coarse = time.perf_counter()
    adj, degree, type_id = _homogeneous_adjacency(data, offsets, type_boundaries)
    cm = _cgc_homo_partition(
        data, offsets, type_boundaries, rep_dict, adj, degree, type_id,
        ratio=ratio, max_hub_degree=max_hub_degree,
        max_candidates=max_candidates)
    t_coarsen = time.perf_counter() - t_coarse

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
        'compressor': 'cgc_homo',
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
        'max_hub_degree': int(max_hub_degree),
        'max_candidates': int(max_candidates),
    }

    if verbose:
        print(f"[cgc_homo] Done: {n_total:,} -> {n_comp:,} nodes  "
              f"(actual ratio={actual_ratio:.3f}, "
              f"{1.0 / max(actual_ratio, 1e-12):.1f}x)  "
              f"total={time.perf_counter() - t_total:.2f}s")

    return HCGCResult(
        data=cdata,
        ratio=actual_ratio,
        node_map=local_cm,
        info=info,
    )


def compress_freehgc(data, ratio=0.1, target_type=None,
                     edge_weight_mode='binary', use_soft_labels=False,
                     freeze_node_types=None, num_hops=2, seed=42,
                     verbose=True):
    """FreeHGC-style training-free heterogeneous data-selection baseline.

    This is an in-pipeline adaptation of FreeHGC's data-selection idea rather
    than a subprocess wrapper around the original code.  It keeps the benchmark
    protocol apples-to-apples by returning the same HCGCResult shape as the
    other compressors:

      1. select target-type training nodes class-wise with a greedy typed
         neighborhood-coverage score;
      2. select other-type representatives by reachability from those target
         representatives;
      3. assign every original node to its nearest selected representative of
         the same type, then reuse the normal quotient-graph builder.

    The requested ratio is interpreted like FreeHGC's reduction rate on target
    training nodes.  The achieved graph compression is therefore reported and
    should be compared by the actual compression column.
    """
    set_seed(seed)
    t_total = time.perf_counter()

    data = _ensure_node_features(data)
    _CFG.node_types = list(data.node_types)
    _CFG.target_type = _detect_target_type(data, target_type)
    _CFG.num_classes = int(data[_CFG.target_type].y.max().item()) + 1
    _CFG.dataset = None

    _, _, _, _, type_boundaries, _, offsets = extract_flat_arrays(
        data, l2_normalize=False)
    n_total = int(type_boundaries[-1])

    t_select = time.perf_counter()
    adj = _global_undirected_adjacency(data, offsets, n_total)
    rep_dict = _freehgc_representations(data, num_hops=num_hops)
    selected = _freehgc_select_representatives(
        data, offsets, type_boundaries, adj, ratio, num_hops, seed)
    cm = _assignment_map_from_representatives(
        data, offsets, type_boundaries, rep_dict, selected)
    t_coarsen = time.perf_counter() - t_select

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
        'compressor': 'freehgc',
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
        'num_hops': int(num_hops),
        'selected_per_type': {nt: int(len(v)) for nt, v in selected.items()},
    }

    if verbose:
        sel = ", ".join(f"{nt}={len(selected.get(nt, []))}"
                        for nt in _CFG.node_types)
        print(f"[freehgc] selected: {sel}")
        print(f"[freehgc] Done: {n_total:,} -> {n_comp:,} nodes  "
              f"(actual ratio={actual_ratio:.3f}, "
              f"{1.0 / max(actual_ratio, 1e-12):.1f}x)  "
              f"total={time.perf_counter() - t_total:.2f}s")

    return HCGCResult(
        data=cdata,
        ratio=actual_ratio,
        node_map=local_cm,
        info=info,
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


def _global_undirected_adjacency(data, offsets, n_total):
    adj = [set() for _ in range(n_total)]
    for et in data.edge_types:
        s_type, _, d_type = et
        s_off = int(offsets[s_type])
        d_off = int(offsets[d_type])
        ei = data[et].edge_index.detach().cpu()
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            u = s_off + int(s)
            v = d_off + int(d)
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
    return [np.fromiter(sorted(v), dtype=np.int64) for v in adj]


def _freehgc_representations(data, num_hops=2, max_dim=512, seed=42):
    """Return per-type propagated features for representative assignment."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)

    h = {}
    for nt in _CFG.node_types:
        h[nt] = _node_representation(data, nt).float()

    chunks = {nt: [h[nt]] for nt in _CFG.node_types}
    cur = {nt: h[nt] for nt in _CFG.node_types}

    for _ in range(max(0, int(num_hops))):
        sums = {nt: torch.zeros_like(cur[nt]) for nt in _CFG.node_types}
        counts = {nt: 0 for nt in _CFG.node_types}
        for et in data.edge_types:
            s_type, _, d_type = et
            if s_type not in cur or d_type not in cur:
                continue
            src = cur[s_type]
            dst_dim = cur[d_type].shape[1]
            if src.shape[1] != dst_dim:
                d = min(src.shape[1], dst_dim)
                msg_src = src[:, :d]
                if d < dst_dim:
                    pad = torch.zeros(src.shape[0], dst_dim - d)
                    msg_src = torch.cat([msg_src, pad], dim=1)
            else:
                msg_src = src
            ei = data[et].edge_index.detach().cpu()
            dst_n = data[d_type].num_nodes
            out = torch.zeros(dst_n, dst_dim)
            deg = torch.zeros(dst_n, 1)
            out.scatter_add_(0, ei[1].unsqueeze(1).expand(-1, dst_dim),
                             msg_src[ei[0]])
            deg.scatter_add_(0, ei[1].unsqueeze(1),
                             torch.ones(ei.shape[1], 1))
            sums[d_type] += out / deg.clamp(min=1.0)
            counts[d_type] += 1

        nxt = {}
        for nt in _CFG.node_types:
            if counts[nt] > 0:
                nxt[nt] = F.normalize(sums[nt] / counts[nt], p=2, dim=1)
            else:
                nxt[nt] = cur[nt]
            chunks[nt].append(nxt[nt])
        cur = nxt

    rep = {}
    for nt, parts in chunks.items():
        x = torch.cat(parts, dim=1).contiguous()
        if x.shape[1] > max_dim:
            proj = torch.randn(x.shape[1], max_dim, generator=gen) / math.sqrt(max_dim)
            x = x @ proj
        rep[nt] = F.normalize(x, p=2, dim=1)
    return rep


def _freehgc_select_representatives(data, offsets, type_boundaries, adj,
                                    ratio, num_hops, seed):
    rng = np.random.default_rng(seed)
    target = _CFG.target_type
    selected = {}

    train_mask = data[target].train_mask.detach().cpu().bool()
    y = data[target].y.detach().cpu().long()
    train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
    labels_train = y[train_mask].numpy()

    n_train = len(train_idx)
    target_budget = max(1, int(math.ceil(float(ratio) * n_train)))
    target_budget = min(target_budget, n_train)
    class_budget = _class_balanced_budget(labels_train, target_budget)
    signatures = _coverage_signatures(
        adj, int(offsets[target]), train_idx, max_hops=num_hops,
        max_items=512)

    target_sel = []
    for cls, cnt in class_budget.items():
        cls_nodes = train_idx[labels_train == cls]
        if len(cls_nodes) == 0:
            continue
        target_sel.extend(_greedy_coverage_select(
            cls_nodes, signatures, cnt, rng))
    if not target_sel and len(train_idx) > 0:
        target_sel = [int(train_idx[0])]
    selected[target] = np.array(sorted(set(int(v) for v in target_sel)), dtype=np.int64)

    target_start = int(offsets[target])
    selected_global = [target_start + int(v) for v in selected[target]]
    reach_scores = _reachability_scores(adj, selected_global, max_hops=num_hops)

    n_target = int(type_boundaries[_CFG.node_types.index(target)]) - target_start
    real_rate = len(selected[target]) / max(n_target, 1)
    for i, nt in enumerate(_CFG.node_types):
        if nt == target:
            continue
        start = int(offsets[nt])
        end = int(type_boundaries[i])
        n = end - start
        if n <= 0:
            selected[nt] = np.empty(0, dtype=np.int64)
            continue
        k = max(1, int(math.ceil(real_rate * n)))
        k = min(k, n)
        scores = np.array([reach_scores.get(start + j, 0.0) for j in range(n)])
        if np.all(scores <= 0):
            local = np.arange(n)
            rng.shuffle(local)
            chosen = np.sort(local[:k])
        else:
            chosen = np.argsort(-scores, kind='stable')[:k]
            chosen = np.sort(chosen)
        selected[nt] = chosen.astype(np.int64)
    return selected


def _class_balanced_budget(labels, total_budget):
    counts = defaultdict(int)
    for c in labels.tolist():
        counts[int(c)] += 1
    if not counts:
        return {}
    items = sorted(counts.items(), key=lambda kv: kv[1])
    budget = {}
    used = 0
    n = max(1, len(labels))
    for idx, (cls, cnt) in enumerate(items):
        if idx == len(items) - 1:
            b = total_budget - used
        else:
            b = max(1, int(math.floor(total_budget * cnt / n)))
            used += b
        budget[cls] = max(0, min(int(cnt), int(b)))
    short = total_budget - sum(budget.values())
    if short > 0:
        for cls, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
            add = min(short, cnt - budget.get(cls, 0))
            budget[cls] = budget.get(cls, 0) + add
            short -= add
            if short <= 0:
                break
    return {c: b for c, b in budget.items() if b > 0}


def _coverage_signatures(adj, type_start, local_nodes, max_hops=2, max_items=512):
    signatures = {}
    for local in local_nodes:
        root = type_start + int(local)
        seen = {root}
        frontier = {root}
        for _ in range(max(1, int(max_hops))):
            nxt = set()
            for u in frontier:
                for v in adj[u]:
                    iv = int(v)
                    if iv not in seen:
                        nxt.add(iv)
            seen.update(nxt)
            frontier = nxt
            if len(seen) >= max_items:
                break
        if len(seen) > max_items:
            signatures[int(local)] = set(sorted(seen)[:max_items])
        else:
            signatures[int(local)] = seen
    return signatures


def _greedy_coverage_select(nodes, signatures, k, rng):
    nodes = [int(v) for v in nodes]
    if k >= len(nodes):
        return nodes
    covered = set()
    remaining = set(nodes)
    selected = []
    degree = {v: len(signatures.get(v, ())) for v in nodes}
    for _ in range(k):
        best = None
        best_score = -1
        for v in remaining:
            sig = signatures.get(v, set())
            score = len(sig - covered) + 1e-3 * degree.get(v, 0)
            if score > best_score or (
                    score == best_score and (best is None or v < best)):
                best = v
                best_score = score
        if best is None:
            best = int(rng.choice(list(remaining)))
        selected.append(best)
        covered.update(signatures.get(best, set()))
        remaining.remove(best)
    return selected


def _reachability_scores(adj, selected_global, max_hops=2):
    scores = defaultdict(float)
    frontier = set(int(v) for v in selected_global)
    visited = set(frontier)
    decay = 1.0
    for u in frontier:
        scores[u] += decay
    for _ in range(max(1, int(max_hops))):
        nxt = set()
        decay *= 0.5
        for u in frontier:
            for v in adj[u]:
                iv = int(v)
                scores[iv] += decay
                if iv not in visited:
                    nxt.add(iv)
                    visited.add(iv)
        frontier = nxt
        if not frontier:
            break
    return scores


def _assignment_map_from_representatives(data, offsets, type_boundaries,
                                         rep_dict, selected):
    n_total = int(type_boundaries[-1])
    cm = np.empty(n_total, dtype=np.int64)

    for i, nt in enumerate(_CFG.node_types):
        start = int(offsets[nt])
        end = int(type_boundaries[i])
        n = end - start
        anchors = np.asarray(selected.get(nt, []), dtype=np.int64)
        anchors = anchors[(anchors >= 0) & (anchors < n)]
        if len(anchors) == 0:
            anchors = np.array([0], dtype=np.int64)

        rep = rep_dict[nt].float()
        anchor_rep = rep[torch.from_numpy(anchors)].float()
        assigned = np.empty(n, dtype=np.int64)
        chunk = 4096
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            score = rep[s:e] @ anchor_rep.t()
            nearest = score.argmax(dim=1).cpu().numpy()
            assigned[s:e] = anchors[nearest]
        cm[start:end] = start + assigned
    return cm


def _homogeneous_adjacency(data, offsets, type_boundaries):
    n_total = int(type_boundaries[-1])
    adj_sets = [set() for _ in range(n_total)]
    type_id = np.empty(n_total, dtype=np.int32)

    for i, nt in enumerate(_CFG.node_types):
        start = int(offsets[nt])
        end = int(type_boundaries[i])
        type_id[start:end] = i

    for et in data.edge_types:
        s_type, _, d_type = et
        s_off = int(offsets[s_type])
        d_off = int(offsets[d_type])
        ei = data[et].edge_index.cpu().numpy()
        for s, d in zip(ei[0], ei[1]):
            u = s_off + int(s)
            v = d_off + int(d)
            if u == v:
                continue
            adj_sets[u].add(v)
            adj_sets[v].add(u)

    adj = [np.fromiter(sorted(a), dtype=np.int64) for a in adj_sets]
    degree = np.array([len(a) for a in adj], dtype=np.int32)
    return adj, degree, type_id


def _cgc_homo_partition(data, offsets, type_boundaries, rep_dict,
                        adj, degree, type_id, ratio=0.1,
                        max_hub_degree=512, max_candidates=128):
    n_total = int(type_boundaries[-1])
    cm = np.arange(n_total, dtype=np.int64)

    for t, nt in enumerate(_CFG.node_types):
        start = int(offsets[nt])
        end = int(type_boundaries[t])
        n = end - start
        if n <= 1:
            continue

        target = max(1, int(math.ceil(float(ratio) * n)))
        rep = rep_dict[nt].float().cpu().numpy().astype(np.float32)
        candidates = _same_type_two_hop_candidates(
            start, end, t, adj, degree, type_id,
            max_hub_degree=max_hub_degree,
            max_candidates=max_candidates)
        local_roots = _cgc_homo_greedy_type(
            rep, degree[start:end], candidates, target)
        cm[start:end] = start + local_roots

    return cm


def _same_type_two_hop_candidates(start, end, t, adj, degree, type_id,
                                  max_hub_degree=512, max_candidates=128):
    n = end - start
    out = []
    max_hub_degree = int(max_hub_degree)
    max_candidates = int(max_candidates)

    for local_u in range(n):
        u = start + local_u
        cand = set()

        for nb in adj[u]:
            nb = int(nb)
            if start <= nb < end:
                cand.add(nb - start)

            if max_hub_degree > 0 and degree[nb] > max_hub_degree:
                continue
            for w in adj[nb]:
                w = int(w)
                if w == u:
                    continue
                if type_id[w] == t:
                    cand.add(w - start)

        cand.discard(local_u)
        if max_candidates > 0 and len(cand) > max_candidates:
            ranked = sorted(cand, key=lambda x: (-int(degree[start + x]), x))
            cand = set(ranked[:max_candidates])
        out.append(np.fromiter(sorted(cand), dtype=np.int64))
    return out


def _cgc_homo_greedy_type(rep, degree_local, candidates, target):
    n = rep.shape[0]
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int32)
    feat_sum = rep.copy()
    active = n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def merge(dst, src):
        nonlocal active
        if dst == src:
            return False
        parent[src] = dst
        feat_sum[dst] += feat_sum[src]
        size[dst] += size[src]
        size[src] = 0
        active -= 1
        return True

    def ward_cost(a, b):
        na = max(int(size[a]), 1)
        nb = max(int(size[b]), 1)
        ma = feat_sum[a] / float(na)
        mb = feat_sum[b] / float(nb)
        diff = ma - mb
        return (na * nb / max(na + nb, 1.0)) * float(np.dot(diff, diff))

    order = np.argsort(-degree_local, kind='stable')
    max_outer = 10
    for _ in range(max_outer):
        changed = False
        for u0 in order:
            if active <= target:
                break
            ru = find(int(u0))
            if size[ru] <= 0:
                continue

            best_r = -1
            best_cost = float('inf')
            seen = set()
            for v0 in candidates[int(u0)]:
                rv = find(int(v0))
                if rv == ru or rv in seen or size[rv] <= 0:
                    continue
                seen.add(rv)
                cost = ward_cost(ru, rv)
                if cost < best_cost:
                    best_cost = cost
                    best_r = rv

            if best_r >= 0 and merge(ru, best_r):
                changed = True
        if active <= target or not changed:
            break

    for i in range(n):
        parent[i] = find(i)
    return parent


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
