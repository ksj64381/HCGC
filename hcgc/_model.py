"""
hcgc/_model.py -- Heterogeneous GNN models (HeteroSAGE, HGT, RGCN),
                  training, and embedding extraction.
"""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):
        return it

import hcgc._config as _cfg
from hcgc._config import _CFG, set_seed

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

MINI_BATCH_THRESHOLD = 100_000  # total nodes; graphs above this use mini-batch


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class HeteroSAGE(nn.Module):
    def __init__(self, edge_types, feat_dims_dict, hidden, num_classes,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.proj = nn.ModuleDict({
            nt.replace('.', '_'): nn.Linear(d, hidden)
            for nt, d in feat_dims_dict.items()
        })
        import inspect as _i
        kw = dict(add_self_loops=False) if 'add_self_loops' in _i.signature(
            SAGEConv.__init__).parameters else {}

        def _make():
            return HeteroConv(
                {et: SAGEConv(hidden, hidden, **kw) for et in edge_types}, aggr='mean')

        self.convs      = nn.ModuleList([_make() for _ in range(num_layers)])
        self.clf        = nn.Linear(hidden, num_classes)
        self.drop       = nn.Dropout(dropout)
        self._ntypes    = list(feat_dims_dict.keys())
        self.num_layers = num_layers

    def encode(self, x_dict, edge_index_dict):
        """Return intermediate embeddings (all node types, before clf head)."""
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return h  # dict[node_type -> Tensor]

    def forward(self, x_dict, edge_index_dict):
        h = self.encode(x_dict, edge_index_dict)
        target_h = h.get(_CFG.target_type)
        if target_h is None:
            raise RuntimeError(f"No embeddings for target type {_CFG.target_type!r}")
        return self.clf(target_h)


# ══════════════════════════════════════════════════════════════════════════════
# HGT (Heterogeneous Graph Transformer)  -- Hu et al., WWW 2020
# ══════════════════════════════════════════════════════════════════════════════

class HGTModel(nn.Module):
    """Heterogeneous Graph Transformer with type-dependent attention.

    Wraps PyG's HGTConv behind the same encode/forward interface as HeteroSAGE
    so it is a drop-in replacement for build_model / eval_pipeline.
    """
    def __init__(self, data, hidden, num_classes, dropout=0.5, num_layers=2, num_heads=2):
        super().__init__()
        from torch_geometric.nn import HGTConv

        self._ntypes = [nt for nt in data.node_types
                        if hasattr(data[nt], 'x') and data[nt].x is not None]
        self.proj = nn.ModuleDict({
            nt.replace('.', '_'): nn.Linear(data[nt].x.shape[1], hidden)
            for nt in self._ntypes
        })
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, data.metadata(), num_heads)
            for _ in range(num_layers)
        ])
        self.clf     = nn.Linear(hidden, num_classes)
        self.drop    = nn.Dropout(dropout)
        self.num_layers = num_layers

    def encode(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            h = {k: (F.relu(self.drop(h_new[k])) if (h_new.get(k) is not None
                     and i < self.num_layers - 1)
                     else (h_new[k] if h_new.get(k) is not None else h[k]))
                 for k in h}
        return h

    def forward(self, x_dict, edge_index_dict):
        h = self.encode(x_dict, edge_index_dict)
        target_h = h.get(_CFG.target_type)
        if target_h is None:
            raise RuntimeError(f"No embeddings for target type {_CFG.target_type!r}")
        return self.clf(target_h)


# ══════════════════════════════════════════════════════════════════════════════
# RGCN (Relational Graph Convolutional Network)  -- Schlichtkrull et al., 2018
# ══════════════════════════════════════════════════════════════════════════════

class RGCNModel(nn.Module):
    """R-GCN style model via HeteroConv + GCNConv per relation type."""
    def __init__(self, data, hidden, num_classes, dropout=0.5, num_layers=2):
        super().__init__()
        from torch_geometric.nn import GraphConv

        self._ntypes = [nt for nt in data.node_types
                        if hasattr(data[nt], 'x') and data[nt].x is not None]
        self.proj = nn.ModuleDict({
            nt.replace('.', '_'): nn.Linear(data[nt].x.shape[1], hidden)
            for nt in self._ntypes
        })

        def _make():
            return HeteroConv(
                {et: GraphConv(hidden, hidden)
                 for et in data.edge_types},
                aggr='sum')

        self.convs   = nn.ModuleList([_make() for _ in range(num_layers)])
        self.clf     = nn.Linear(hidden, num_classes)
        self.drop    = nn.Dropout(dropout)
        self.num_layers = num_layers

    def encode(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return h

    def forward(self, x_dict, edge_index_dict):
        h = self.encode(x_dict, edge_index_dict)
        target_h = h.get(_CFG.target_type)
        if target_h is None:
            raise RuntimeError(f"No embeddings for target type {_CFG.target_type!r}")
        return self.clf(target_h)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _x_dict(hdata):
    return {nt: hdata[nt].x.to(_cfg.DEVICE)
            for nt in hdata.node_types
            if hasattr(hdata[nt], 'x') and hdata[nt].x is not None}


def _ei_dict(hdata):
    return {et: hdata[et].edge_index.to(_cfg.DEVICE) for et in hdata.edge_types}


_GNN_MODELS = ('sage', 'hgt', 'rgcn')


def build_model(hdata, hidden, dropout, num_layers=2, gnn_model='sage'):
    """Build a heterogeneous GNN.

    Args:
        gnn_model: one of 'sage' (HeteroSAGE, default), 'hgt' (HGT), 'rgcn' (R-GCN).
    """
    gnn_model = gnn_model.lower()
    if gnn_model == 'hgt':
        return HGTModel(hdata, hidden, _CFG.num_classes,
                        dropout, num_layers).to(_cfg.DEVICE)
    elif gnn_model == 'rgcn':
        return RGCNModel(hdata, hidden, _CFG.num_classes,
                         dropout, num_layers).to(_cfg.DEVICE)
    else:  # 'sage' (default)
        fdims = {nt: hdata[nt].x.shape[1]
                 for nt in hdata.node_types
                 if hasattr(hdata[nt], 'x') and hdata[nt].x is not None}
        return HeteroSAGE(hdata.edge_types, fdims, hidden,
                          _CFG.num_classes, dropout, num_layers).to(_cfg.DEVICE)


def _acc(out, y, mask):
    return (out[mask].argmax(1) == y[mask]).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# Full-batch training
# ══════════════════════════════════════════════════════════════════════════════

def train_full_batch(model, hdata, epochs, lr, desc='',
                     eval_every=10, patience=60, use_soft_labels=False):
    xd  = _x_dict(hdata)
    eid = _ei_dict(hdata)
    y   = hdata[_CFG.target_type].y.to(_cfg.DEVICE)
    tr  = hdata[_CFG.target_type].train_mask.to(_cfg.DEVICE)
    va  = hdata[_CFG.target_type].val_mask.to(_cfg.DEVICE)
    te  = hdata[_CFG.target_type].test_mask.to(_cfg.DEVICE)

    soft_y = None
    if use_soft_labels and hasattr(hdata[_CFG.target_type], 'soft_y'):
        soft_y = hdata[_CFG.target_type].soft_y.to(_cfg.DEVICE)
        print(f"  [{desc}] Soft-label cross-entropy ON")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best  = {'val': 0.0, 'test': 0.0, 'epoch': 0}
    no_imp = 0
    best_state = None

    t0 = time.time()
    ep_bar = _tqdm(range(1, epochs + 1), desc=f'  [{desc}]', unit='ep',
                   ncols=88, leave=True)
    for ep in ep_bar:
        model.train(); opt.zero_grad()
        out = model(xd, eid)
        if soft_y is not None:
            loss = -(soft_y[tr] * F.log_softmax(out[tr], dim=1)).sum(1).mean()
        else:
            loss = F.cross_entropy(out[tr], y[tr], ignore_index=-1)
        loss.backward(); opt.step(); sched.step()

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out = model(xd, eid)
            va_acc = _acc(out, y, va)
            te_acc = _acc(out, y, te)
            val_is_nan = math.isnan(va_acc)
            if val_is_nan:
                if te_acc > best['test']:
                    best.update({'val': float('nan'), 'test': te_acc, 'epoch': ep})
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                val_str = 'nan'
                best_val_str = 'nan'
            else:
                if va_acc > best['val']:
                    best.update({'val': va_acc, 'test': te_acc, 'epoch': ep}); no_imp = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_imp += eval_every
                val_str = f'{va_acc:.4f}'
                best_val_str = f'{best["val"]:.4f}'
            ep_bar.set_postfix(
                loss=f'{loss.item():.4f}',
                val=val_str,
                best=best_val_str,
                pat=no_imp,
            )
            if not val_is_nan and no_imp >= patience:
                ep_bar.set_description(f'  [{desc}][early-stop]')
                break
        else:
            ep_bar.set_postfix(loss=f'{loss.item():.4f}', best=f'{best["val"]:.4f}')

    # Restore best-val weights
    if best_state is not None:
        model.load_state_dict({k: v.to(_cfg.DEVICE) for k, v in best_state.items()})

    print(f"  [{desc}] Done {time.time()-t0:.1f}s - best val={best['val']:.4f} "
          f"test={best['test']:.4f}")
    return best, model


# ══════════════════════════════════════════════════════════════════════════════
# Mini-batch training (NeighborLoader)
# ══════════════════════════════════════════════════════════════════════════════

def _default_num_neighbors(data):
    # [5, 3] keeps subgraphs manageable on dense hetero graphs (e.g. ogbn-mag
    # 7 edge types). Matches the downstream training setting in benchmark.py.
    return {et: [5, 3] for et in data.edge_types}


def train_mini_batch(model, data, epochs, lr,
                     batch_size=512, num_neighbors=None,
                     desc='', eval_every=10, patience=60,
                     use_soft_labels=False):
    """Mini-batch training via NeighborLoader."""
    from torch_geometric.loader import NeighborLoader

    tt = _CFG.target_type
    if num_neighbors is None:
        num_neighbors = _default_num_neighbors(data)

    import os as _os
    _n_workers = 4 if _os.name == 'posix' else 0

    train_mask = data[tt].train_mask
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(tt, train_mask),
        shuffle=True,
        num_workers=_n_workers,
    )

    soft_y = None
    if use_soft_labels and hasattr(data[tt], 'soft_y'):
        soft_y = data[tt].soft_y
        print(f"  [{desc}] Soft-label cross-entropy ON (mini-batch)")

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best  = {'val': 0.0, 'test': 0.0, 'epoch': 0}
    no_imp = 0
    best_state = None

    t0 = time.time()
    ep_bar = _tqdm(range(1, epochs + 1), desc=f'  [{desc}]', unit='ep',
                   ncols=88, leave=True)
    for ep in ep_bar:
        model.train()
        ep_loss = 0.0; n_batches = 0
        for batch in train_loader:
            batch = batch.to(_cfg.DEVICE)
            opt.zero_grad()
            out = model(_x_dict(batch), _ei_dict(batch))
            bs  = batch[tt].batch_size
            b_mask = batch[tt].train_mask[:bs]
            b_y    = batch[tt].y[:bs]
            if b_mask.sum() == 0:
                continue
            if soft_y is not None:
                b_nids  = batch[tt].n_id[:bs].cpu()
                b_soft  = soft_y[b_nids][b_mask.cpu()].to(_cfg.DEVICE)
                loss = -(b_soft * F.log_softmax(out[:bs][b_mask], dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(out[:bs][b_mask], b_y[b_mask],
                                       ignore_index=-1)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)

        if ep % eval_every == 0 or ep == epochs:
            va_acc, te_acc = _eval_mini_batch(model, data, batch_size, num_neighbors)
            val_is_nan = math.isnan(va_acc)
            if val_is_nan:
                if te_acc > best['test']:
                    best.update({'val': float('nan'), 'test': te_acc, 'epoch': ep})
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                val_str = best_val_str = 'nan'
            else:
                if va_acc > best['val']:
                    best.update({'val': va_acc, 'test': te_acc, 'epoch': ep})
                    no_imp = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_imp += eval_every
                val_str      = f'{va_acc:.4f}'
                best_val_str = f'{best["val"]:.4f}'
            ep_bar.set_postfix(
                loss=f'{avg_loss:.4f}',
                val=val_str,
                best=best_val_str,
                pat=no_imp,
            )
            if not val_is_nan and no_imp >= patience:
                ep_bar.set_description(f'  [{desc}][early-stop]')
                break
        else:
            ep_bar.set_postfix(loss=f'{avg_loss:.4f}', best=f'{best["val"]:.4f}')

    # Restore best-val weights
    if best_state is not None:
        model.load_state_dict({k: v.to(_cfg.DEVICE) for k, v in best_state.items()})

    print(f"  [{desc}] Done {time.time()-t0:.1f}s - best val={best['val']:.4f} "
          f"test={best['test']:.4f}")
    return best, model


def _eval_mini_batch(model, data, batch_size=512, num_neighbors=None):
    """Inference over the full graph using NeighborLoader; returns (val_acc, test_acc)."""
    from torch_geometric.loader import NeighborLoader

    tt = _CFG.target_type
    if num_neighbors is None:
        num_neighbors = _default_num_neighbors(data)

    import os as _os
    _n_workers = 4 if _os.name == 'posix' else 0

    n_nodes = data[tt].num_nodes
    all_nodes_mask = torch.ones(n_nodes, dtype=torch.bool)
    inf_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=(tt, all_nodes_mask),
        shuffle=False,
        num_workers=_n_workers,
    )

    model.eval()
    preds = torch.empty(n_nodes, dtype=torch.long)
    with torch.no_grad():
        for batch in inf_loader:
            batch = batch.to(_cfg.DEVICE)
            out = model(_x_dict(batch), _ei_dict(batch))
            bs   = batch[tt].batch_size
            pred = out[:bs].argmax(1).cpu()
            nids = batch[tt].n_id[:bs].cpu()
            preds[nids] = pred

    y       = data[tt].y.cpu()
    va_mask = data[tt].val_mask.cpu()
    te_mask = data[tt].test_mask.cpu()
    va_acc  = (preds[va_mask] == y[va_mask]).float().mean().item() if va_mask.any() else float('nan')
    te_acc  = (preds[te_mask] == y[te_mask]).float().mean().item() if te_mask.any() else float('nan')
    return va_acc, te_acc


def _extract_emb_mini_batch(model, data, batch_size=512, num_neighbors=None):
    """Extract node embeddings (before clf head) using NeighborLoader."""
    from torch_geometric.loader import NeighborLoader

    tt = _CFG.target_type
    if num_neighbors is None:
        num_neighbors = _default_num_neighbors(data)

    node_counts = {nt: data[nt].num_nodes for nt in _CFG.node_types}
    total = sum(node_counts.values())

    if total < MINI_BATCH_THRESHOLD:
        model.eval()
        with torch.no_grad():
            data_dev = data.to(_cfg.DEVICE)
            h = model.encode(_x_dict(data_dev), _ei_dict(data_dev))
        return {nt: h[nt].cpu() for nt in _CFG.node_types if nt in h}

    emb_dim = next(iter(model.proj.values())).out_features
    emb_acc  = {nt: torch.zeros(n, emb_dim) for nt, n in node_counts.items()}
    emb_cnt  = {nt: torch.zeros(n, dtype=torch.long) for nt, n in node_counts.items()}

    # Embedding extraction is inference-only: num_workers=0 avoids the
    # "Too many open files" crash that occurs when iterating over many node
    # types in a loop (each NeighborLoader with workers>0 opens shared-memory
    # handles; with 8 node types × 36 edge types the fd limit is hit quickly).
    for seed_nt in _CFG.node_types:
        n_seed = data[seed_nt].num_nodes
        seed_mask = torch.ones(n_seed, dtype=torch.bool)
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=(seed_nt, seed_mask),
            shuffle=False,
            num_workers=0,
        )
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(_cfg.DEVICE)
                h = model.encode(_x_dict(batch), _ei_dict(batch))
                bs = batch[seed_nt].batch_size
                seed_nids = batch[seed_nt].n_id[:bs].cpu()
                seed_h    = h.get(seed_nt)
                if seed_h is not None:
                    emb_acc[seed_nt][seed_nids] += seed_h[:bs].cpu()
                    emb_cnt[seed_nt][seed_nids] += 1

    result = {}
    for nt in _CFG.node_types:
        cnt = emb_cnt[nt].float().clamp(min=1).unsqueeze(1)
        result[nt] = emb_acc[nt] / cnt
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Auto-select helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_large_graph(data, args=None):
    """Return True if mini-batch mode should be used."""
    if args is not None and getattr(args, 'force_mini_batch', False):
        return True
    return sum(data[nt].num_nodes for nt in data.node_types) >= MINI_BATCH_THRESHOLD


def _train_auto(model, data, epochs, lr, args, desc=''):
    """Auto-select full-batch or mini-batch training based on graph size."""
    if _is_large_graph(data, args):
        nb     = getattr(args, 'mini_batch_size', 512)
        nn_cfg = getattr(args, 'num_neighbors', None)
        print(f"  [{desc}] Large graph -> mini-batch training (batch_size={nb})")
        return train_mini_batch(model, data, epochs, lr,
                                batch_size=nb,
                                num_neighbors=nn_cfg,
                                desc=desc,
                                eval_every=args.eval_every,
                                patience=args.patience,
                                use_soft_labels=args.use_soft_labels)
    else:
        return train_full_batch(model, data, epochs, lr,
                                desc=desc,
                                eval_every=args.eval_every,
                                patience=args.patience,
                                use_soft_labels=args.use_soft_labels)


def _extract_emb_auto(model, data, args):
    """Auto-select embedding extraction method based on graph size."""
    if _is_large_graph(data, args):
        nb     = getattr(args, 'mini_batch_size', 512)
        nn_cfg = getattr(args, 'num_neighbors', None)
        print(f"  [Emb] Large graph -> mini-batch embedding extraction (batch_size={nb})")
        return _extract_emb_mini_batch(model, data, batch_size=nb, num_neighbors=nn_cfg)
    else:
        return extract_embeddings(model, data, mode=args.emb_mode)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(model, data, mode='conv'):
    model.eval()
    xd  = _x_dict(data)
    eid = _ei_dict(data)
    with torch.no_grad():
        h = {nt: F.relu(model.proj[nt.replace('.', '_')](x)) for nt, x in xd.items()}
        if mode == 'conv':
            h_out = model.convs[0](h, eid)
            h = {nt: F.relu(h_out[nt]) if (nt in h_out and h_out[nt] is not None)
                 else h[nt] for nt in h}
    return {nt: h[nt].cpu() for nt in _CFG.node_types if nt in h}


def extract_metapath2vec_embeddings(data, embedding_dim=128,
                                    walk_length=50, context_size=7,
                                    walks_per_node=5, epochs=5, lr=0.01):
    """MetaPath2Vec embeddings as an alternative to GNN pretrain."""
    from torch_geometric.nn import MetaPath2Vec as _M2V

    et     = data.edge_types
    et_set = set(et)
    tt     = _CFG.target_type
    nts    = list(_CFG.node_types)

    metapaths = []
    for nt in nts:
        fwd = (tt, 'to', nt)
        bwd = (nt, 'to', tt)
        if fwd in et_set and bwd in et_set and nt != tt:
            metapaths.append([fwd, bwd])

    if not metapaths:
        for s, r, d in et:
            rev = (d, 'to', s)
            if rev in et_set:
                metapaths.append([(s, r, d), rev])
                break

    print(f"  [MetaPath2Vec] metapaths={metapaths}")

    ei_dict = {et: data[et].edge_index.to(_cfg.DEVICE) for et in data.edge_types}

    emb_dict = {}
    for mp in metapaths:
        model = _M2V(
            ei_dict,
            embedding_dim=embedding_dim,
            metapath=mp,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=1,
            sparse=True,
        ).to(_cfg.DEVICE)

        loader = model.loader(batch_size=512, shuffle=True, num_workers=0)
        opt = torch.optim.SparseAdam(model.parameters(), lr=lr)

        model.train()
        for ep in range(1, epochs + 1):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                opt.zero_grad()
                loss = model.loss(pos_rw.to(_cfg.DEVICE), neg_rw.to(_cfg.DEVICE))
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if ep % max(1, epochs // 3) == 0:
                print(f"  [MetaPath2Vec] ep {ep}/{epochs} loss={total_loss/len(loader):.4f}")

        model.eval()
        with torch.no_grad():
            for nt in nts:
                try:
                    h = model(nt).cpu().detach()
                    if nt not in emb_dict:
                        emb_dict[nt] = h
                    else:
                        emb_dict[nt] = torch.cat([emb_dict[nt], h], dim=1)
                except Exception:
                    pass

    for nt in nts:
        if nt not in emb_dict:
            n = data[nt].num_nodes
            emb_dict[nt] = torch.zeros(n, embedding_dim)

    print(f"  [MetaPath2Vec] Done. embedding shapes: "
          + ", ".join(f"{nt}={emb_dict[nt].shape}" for nt in nts))
    return emb_dict


@torch.no_grad()
def fast_embed_hetero(data, n_hops=2, out_dim=128, device=None, rng_seed=42, verbose=True):
    """Training-free heterogeneous feature propagation (SGC-style).

    Complexity: O(K * |E| * out_dim)  -- no backprop, 10-100x faster than
    GNN pretrain on large graphs.
    """
    if device is None:
        device = _cfg.DEVICE

    t0  = time.time()
    gen = torch.Generator(device='cpu')
    gen.manual_seed(rng_seed)

    node_types = list(data.node_types)
    edge_types = list(data.edge_types)

    h = {}
    has_feat = {}
    for nt in node_types:
        n = data[nt].num_nodes
        x_raw = getattr(data[nt], 'x', None)
        if x_raw is not None and x_raw.numel() > 0:
            x = x_raw.float().to(device)
            d = x.size(1)
            if d == out_dim:
                proj = x
            elif d > out_dim:
                P = torch.randn(d, out_dim, generator=gen).to(device) / (out_dim ** 0.5)
                proj = x @ P
            else:
                pad = torch.zeros(n, out_dim - d, device=device)
                proj = torch.cat([x, pad], dim=1)
            h[nt] = F.normalize(proj, p=2, dim=1)
            has_feat[nt] = True
        else:
            h[nt] = torch.zeros(n, out_dim, device=device)
            has_feat[nt] = False

    if verbose:
        have  = [nt for nt in node_types if has_feat[nt]]
        empty = [nt for nt in node_types if not has_feat[nt]]
        print(f"  [FastEmbed] out_dim={out_dim}  hops={n_hops}")
        if have:  print(f"    feat types  : {have}")
        if empty: print(f"    zero-init   : {empty}  (filled by propagation)")

    adjs = []
    for et in edge_types:
        s_type, _, d_type = et
        ei = data[et].edge_index.to(device)
        if ei.size(1) == 0:
            continue
        n_src = data[s_type].num_nodes
        n_dst = data[d_type].num_nodes
        row = ei[1]
        col = ei[0]

        deg = torch.zeros(n_dst, device=device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
        val  = (1.0 / deg[row].clamp(min=1.0)).float()

        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]), val, (n_dst, n_src), device=device
        ).coalesce()
        adjs.append((s_type, d_type, adj))

    for hop in range(n_hops):
        agg_sum   = {nt: torch.zeros_like(h[nt]) for nt in node_types}
        agg_count = {nt: 0 for nt in node_types}

        for s_type, d_type, adj in adjs:
            src = h[s_type]
            if hop == 0 and not has_feat[s_type]:
                continue
            agg_sum[d_type]    = agg_sum[d_type] + torch.sparse.mm(adj, src)
            agg_count[d_type] += 1

        h_new = {}
        for nt in node_types:
            if agg_count[nt] > 0:
                propagated = agg_sum[nt] / agg_count[nt]
                if has_feat[nt]:
                    h_new[nt] = F.normalize(h[nt] + propagated, p=2, dim=1)
                else:
                    h_new[nt] = F.normalize(propagated, p=2, dim=1)
                    has_feat[nt] = True
            else:
                h_new[nt] = h[nt]
        h = h_new

        if verbose:
            print(f"  [FastEmbed] hop {hop+1}/{n_hops}  ({time.time()-t0:.1f}s)")

    if verbose:
        print(f"  [FastEmbed] Done  {time.time()-t0:.1f}s  "
              + "  ".join(f"{nt}:{h[nt].shape}" for nt in node_types))

    return h


def extract_emb_flat_arrays(emb_dict, l2_normalize=True):
    """Flatten embedding dict to a contiguous float32 array for the C++ core."""
    feats_list, feat_dims = [], []
    for nt in _CFG.node_types:
        x = emb_dict[nt].float()
        if l2_normalize:
            norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            x = x / norms
        x = x.numpy().astype(np.float32)
        feats_list.append(x.ravel())
        feat_dims.append(x.shape[1])
    return np.concatenate(feats_list), np.array(feat_dims, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Eval on original graph
# ══════════════════════════════════════════════════════════════════════════════

def eval_compressed_on_original(comp_model, comp_data, local_cm_target, orig_data):
    comp_model.eval()
    with torch.no_grad():
        out = comp_model(_x_dict(comp_data), _ei_dict(comp_data))
    te_mask   = orig_data[_CFG.target_type].test_mask
    y_orig    = orig_data[_CFG.target_type].y[te_mask]
    super_idx = local_cm_target[te_mask]
    pred = out[super_idx.to(_cfg.DEVICE)].argmax(1).cpu()
    return (pred == y_orig).float().mean().item()
