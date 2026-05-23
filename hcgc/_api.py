"""
hcgc/_api.py -- Public compress() API and HCGCResult dataclass.
"""

import types
import numpy as np
import torch
from dataclasses import dataclass
from torch_geometric.data import HeteroData


@dataclass
class HCGCResult:
    """Result of hcgc.compress().

    Attributes:
        data     : compressed HeteroData graph
        ratio    : actual achieved node retention ratio (n_comp / n_orig)
        node_map : dict {node_type -> LongTensor} mapping original -> supernode index
        info     : dict with detailed stats (compression factor, timings, etc.)
    """
    data:     HeteroData
    ratio:    float
    node_map: dict
    info:     dict


def compress(
    data,
    ratio           = 0.1,
    target_type     = None,
    pretrain        = True,
    pretrain_epochs = 30,
    device          = 'auto',
    verbose         = True,
    mini_batch_size = 512,
    num_neighbors   = None,
) -> HCGCResult:
    """Compress a heterogeneous graph using HCGC.

    Args:
        data            : PyG HeteroData object with node features and edge indices.
                          The target node type must have .y, .train_mask, .val_mask,
                          .test_mask attributes.
        ratio           : Fraction of nodes to keep.
                          0.1 = keep 10% of nodes = 10x compression.
        target_type     : Classification target node type. Auto-detected if None
                          (looks for the node type with train_mask labels).
        pretrain        : If True, pretrain a GNN to get better node embeddings
                          before coarsening. Recommended for higher quality.
                          Graphs with >100k nodes automatically use mini-batch
                          training to avoid GPU OOM.
        pretrain_epochs : Number of pretrain epochs (used when pretrain=True and
                          the graph is small enough for full-batch training).
        device          : Compute device: 'auto', 'cpu', or 'cuda'.
                          'auto' selects CUDA if available.
        verbose         : Print progress messages.
        mini_batch_size : Seed-node batch size for mini-batch training and
                          embedding extraction on large graphs (>100k nodes).
                          Reduce (e.g. 128) if GPU OOM on dense graphs like AMiner.
        num_neighbors   : Neighbours to sample per hop in mini-batch mode.
                          None = auto (10 per hop). Pass a list e.g. [10, 5]
                          to control per-hop sampling and reduce subgraph size.

    Returns:
        HCGCResult with:
            .data     -- compressed HeteroData
            .ratio    -- actual achieved ratio (may differ slightly from requested)
            .node_map -- {node_type: LongTensor} original-node -> supernode mapping
            .info     -- dict with compression factor, timing, and auto-coarsen details

    Example::

        import hcgc
        result = hcgc.compress(data, ratio=0.1)
        print(f"Compressed {result.info['n_nodes_orig']:,} -> "
              f"{result.info['n_nodes_comp']:,} nodes  "
              f"({result.info['compression']:.1f}x)")
        # Use result.data as a regular PyG HeteroData for downstream GNN training
    """
    import hcgc._config as _cfg_mod
    from hcgc._config import _CFG, set_seed
    from hcgc._pipeline import _load_and_pretrain, _coarsen_from_context
    from hcgc._coarsen import build_compressed_data

    # ── Device setup ──────────────────────────────────────────────────────────
    if device == 'auto':
        dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        dev_str = str(device)
    _cfg_mod.set_device(dev_str)

    # ── Ensure all node types have feature tensors ────────────────────────────
    # Some datasets (e.g. DBLP 'conference', ogbn-mag non-paper types) ship
    # without node features.  Inject a 1-D log-degree feature so the pipeline
    # never sees a missing x.
    data = _ensure_node_features(data)

    # ── Configure _CFG from the provided HeteroData ──────────────────────────
    _CFG.node_types  = list(data.node_types)
    _CFG.target_type = _detect_target_type(data, target_type)
    _CFG.num_classes = int(data[_CFG.target_type].y.max().item()) + 1
    _CFG.dataset     = None

    if verbose:
        total = sum(data[nt].num_nodes for nt in _CFG.node_types)
        print(f"[HCGC] target_type={_CFG.target_type!r}  "
              f"num_classes={_CFG.num_classes}  device={dev_str}")
        print(f"[HCGC] nodes: {total:,}  "
              f"ratio={ratio:.3f} ({1/ratio:.1f}x compression target)")

    # ── Build args ────────────────────────────────────────────────────────────
    args = _build_args(ratio=ratio, pretrain=pretrain,
                       pretrain_epochs=pretrain_epochs,
                       mini_batch_size=mini_batch_size,
                       num_neighbors=num_neighbors)

    # ── Pretrain (or fast-embed) + extract flat arrays ────────────────────────
    ctx = _load_and_pretrain(data, args)

    # ── Auto-coarsen ──────────────────────────────────────────────────────────
    cm, t_c = _coarsen_from_context(ctx, args)

    # ── Build compressed HeteroData ───────────────────────────────────────────
    cdata, local_cm, stats = build_compressed_data(
        data, cm,
        ctx['offsets'], ctx['type_boundaries'],
        use_soft_labels=False,
        emb_dict=ctx['emb_dict'] if pretrain else None,
    )

    n_orig = int(ctx['type_boundaries'][-1])
    n_comp = len(np.unique(cm))
    actual_ratio = n_comp / n_orig

    info = {
        'compression':  round(1.0 / actual_ratio, 4),
        'n_nodes_orig': n_orig,
        'n_nodes_comp': n_comp,
        'coarsen_time': round(t_c, 2),
        'nodes_orig':   stats['nodes_orig'],
        'nodes_comp':   stats['nodes_comp'],
        'edges_orig':   stats['edges_orig'],
        'edges_comp':   stats['edges_comp'],
        'edge_ratio':   round(stats['edge_ratio'], 4),
    }

    if verbose:
        print(f"\n[HCGC] Done: {n_orig:,} -> {n_comp:,} nodes  "
              f"(actual ratio={actual_ratio:.3f}, {1/actual_ratio:.1f}x)")

    return HCGCResult(
        data=cdata,
        ratio=actual_ratio,
        node_map=local_cm,
        info=info,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_node_features(data):
    """Inject log-degree features for node types that have no x tensor.

    Some datasets ship without features for certain node types
    (e.g. DBLP 'conference', ogbn-mag author/institution/field_of_study).
    A 1-D log-degree feature is a cheap structural signal that keeps the
    coarsening kernel from crashing on a missing attribute.
    """
    for nt in data.node_types:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            continue
        n = data[nt].num_nodes
        deg = torch.zeros(n, dtype=torch.float)
        for et in data.edge_types:
            _, _, d_type = et
            if d_type == nt:
                ei = data[et].edge_index
                deg.scatter_add_(0, ei[1], torch.ones(ei.shape[1]))
        data[nt].x = (deg + 1.0).log().unsqueeze(1)  # shape [N, 1]
    return data


def _detect_target_type(data, target_type):
    if target_type is not None:
        if target_type not in data.node_types:
            raise ValueError(
                f"target_type={target_type!r} not in data.node_types={list(data.node_types)}"
            )
        return target_type

    # Auto-detect: prefer node types with y + train_mask
    candidates = []
    for nt in data.node_types:
        if hasattr(data[nt], 'y') and data[nt].y is not None:
            if hasattr(data[nt], 'train_mask') and data[nt].train_mask is not None:
                candidates.append(nt)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        return max(candidates, key=lambda nt: data[nt].train_mask.sum().item())

    # Fallback: any node type with y
    for nt in data.node_types:
        if hasattr(data[nt], 'y') and data[nt].y is not None:
            return nt

    raise ValueError(
        "Could not auto-detect target_type. Please specify target_type explicitly. "
        f"Available node types: {list(data.node_types)}"
    )


def _build_args(ratio, pretrain, pretrain_epochs,
                mini_batch_size=512, num_neighbors=None):
    """Build default args namespace for compress()."""
    return types.SimpleNamespace(
        # Coarsening
        coarsen_method              = 'hcgc',
        use_auto_coarsen            = True,
        target_ratio                = ratio,
        max_acc_loss                = 0.1,
        use_fast_scale              = False,
        coarsen_pca_dim             = 0,
        # HCGC kernel params
        hcgc_inner_passes           = 2,
        hcgc_max_outer              = 10,
        hcgc_feat_var_scale         = 1.0,
        hcgc_feat_var_scale_by_type = None,
        hcgc_skip_reassignment      = False,
        hcgc_window_size            = 20,
        hcgc_target_comp_ratio      = 0.0,
        hcgc_num_levels             = 5,
        hub_anchor_percentile       = 0.0,
        max_candidates              = 5,
        auto_hub_caps               = True,
        hub_degree_caps             = '',
        num_levels                  = 5,
        # Pretrain / embedding
        use_emb_coarsen             = pretrain,
        emb_method                  = 'gnn',
        pretrain_epochs             = pretrain_epochs,
        pretrain_patience           = 0,
        pretrain_hidden             = None,
        emb_mode                    = 'conv',
        fast_embed                  = False,
        fast_embed_hops             = 2,
        fast_embed_outdim           = 128,
        # GNN architecture
        hidden                      = 256,
        dropout                     = 0.5,
        num_layers                  = 2,
        gnn_model                   = 'sage',
        # Training
        lr                          = 0.001,
        epochs                      = 200,
        eval_every                  = 10,
        patience                    = 30,
        use_soft_labels             = False,
        emb_temp                    = 1.0,
        # Graph size
        mini_batch_size             = mini_batch_size,
        force_mini_batch            = False,
        num_neighbors               = num_neighbors,
        # Misc
        base_seed                   = 42,
        use_label_aware_split       = False,
    )
