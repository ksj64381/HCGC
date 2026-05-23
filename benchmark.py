#!/usr/bin/env python
"""
benchmark.py -- Reproduce HCGC compression benchmarks.

Supported datasets: imdb, dblp, lastfm, ogbn-mag

Timing note
-----------
The first call to the C++ kernel (hcgc_module) and the first PyTorch backward
pass both incur JIT / CUDA-warmup overhead that can dominate short runs.
This script excludes warmup runs from all reported measurements; see --warmup.

Usage
-----
    python benchmark.py --dataset imdb    --ratio 0.1
    python benchmark.py --dataset dblp    --ratio 0.1 --runs 5
    python benchmark.py --dataset lastfm  --ratio 0.1
    python benchmark.py --dataset ogbn-mag --ratio 0.1 --runs 1 --warmup 1
"""

import argparse
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

import hcgc


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

def _add_degree_features(data):
    """Inject log-degree features for node types that have no x tensor.

    Used for ogbn-mag where only 'paper' nodes have pre-computed features.
    A single log-degree feature is a cheap but informative structural signal.
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
        data[nt].x = (deg + 1.0).log().unsqueeze(1)   # shape [N, 1]
    return data


def load_imdb(root):
    from torch_geometric.datasets import IMDB
    data = IMDB(root=f'{root}/IMDB')[0]
    return data, 'movie'


def load_dblp(root):
    from torch_geometric.datasets import DBLP
    data = DBLP(root=f'{root}/DBLP')[0]
    return data, 'author'


def load_lastfm(root):
    from torch_geometric.datasets import LastFM
    data = LastFM(root=f'{root}/LastFM')[0]
    return data, 'user'


def load_ogbn_mag(root):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        sys.exit("ogbn-mag requires the ogb package:  pip install ogb")

    dataset   = PygNodePropPredDataset(name='ogbn-mag', root=f'{root}/ogbn-mag')
    data      = dataset[0]
    split_idx = dataset.get_idx_split()

    # Convert split index dictionaries → boolean masks on 'paper' nodes
    n = data['paper'].num_nodes
    for split, attr in [('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask')]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[split_idx[split]['paper']] = True
        setattr(data['paper'], attr, mask)

    if data['paper'].y.dim() == 2:
        data['paper'].y = data['paper'].y.squeeze(1)

    # author / institution / field_of_study have no features → add log-degree
    data = _add_degree_features(data)
    return data, 'paper'


LOADERS = {
    'imdb':     load_imdb,
    'dblp':     load_dblp,
    'lastfm':   load_lastfm,
    'ogbn-mag': load_ogbn_mag,
}


# ══════════════════════════════════════════════════════════════════════════════
# Downstream GNN (HeteroSAGE, self-contained — no internal hcgc imports)
# ══════════════════════════════════════════════════════════════════════════════

class _HeteroSAGE(torch.nn.Module):
    """Two-layer HeteroSAGE used for downstream evaluation on compressed graphs."""

    def __init__(self, edge_types, feat_dims, hidden, num_classes, dropout=0.5):
        super().__init__()
        import inspect
        kw = {'add_self_loops': False} if 'add_self_loops' in \
            inspect.signature(SAGEConv.__init__).parameters else {}

        self.proj = torch.nn.ModuleDict({
            nt.replace('.', '_'): torch.nn.Linear(d, hidden)
            for nt, d in feat_dims.items()
        })
        def _conv():
            return HeteroConv(
                {et: SAGEConv(hidden, hidden, **kw) for et in edge_types},
                aggr='mean')
        self.conv1 = _conv()
        self.conv2 = _conv()
        self.clf   = torch.nn.Linear(hidden, num_classes)
        self.drop  = torch.nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, target_type):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items()
             if nt.replace('.', '_') in self.proj}
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        h = self.conv2(h, edge_index_dict)
        return self.clf(h[target_type])


def train_on_heterodata(data, target_type, device_str,
                        epochs=200, lr=1e-3, patience=30, hidden=256):
    """Train a 2-layer HeteroSAGE on any HeteroData object.

    Returns
    -------
    test_acc : float
    elapsed  : float   wall-clock seconds (model init + all epochs)
    """
    dev = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_str == 'auto' else device_str)
    cdata = data.to(dev)

    feat_dims = {
        nt: cdata[nt].x.shape[1]
        for nt in cdata.node_types
        if hasattr(cdata[nt], 'x') and cdata[nt].x is not None
    }
    num_classes = int(cdata[target_type].y.max().item()) + 1

    model = _HeteroSAGE(cdata.edge_types, feat_dims, hidden, num_classes).to(dev)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_test = 0.0, 0.0
    no_improve          = 0          # measured in eval steps, not epochs
    eval_every          = 10

    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out  = model(cdata.x_dict, cdata.edge_index_dict, target_type)
        mask = cdata[target_type].train_mask
        loss = F.cross_entropy(out[mask], cdata[target_type].y[mask])
        loss.backward()
        opt.step()

        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                out = model(cdata.x_dict, cdata.edge_index_dict, target_type)
            pred = out.argmax(dim=1)
            y    = cdata[target_type].y

            val_acc  = (pred[cdata[target_type].val_mask]
                        == y[cdata[target_type].val_mask]).float().mean().item()
            test_acc = (pred[cdata[target_type].test_mask]
                        == y[cdata[target_type].test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val, best_test, no_improve = val_acc, test_acc, 0
            else:
                no_improve += 1

            if no_improve >= patience // eval_every:
                break

    return best_test, time.perf_counter() - t0


def train_downstream(result, target_type, device_str,
                     epochs=200, lr=1e-3, patience=30, hidden=256):
    """Thin wrapper: train on a compressed HCGCResult."""
    return train_on_heterodata(result.data, target_type, device_str,
                               epochs=epochs, lr=lr, patience=patience,
                               hidden=hidden)


# ══════════════════════════════════════════════════════════════════════════════
# Single benchmark run
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(data, target_type, device, train_epochs=200, train_hidden=256):
    """Train on the original (uncompressed) graph. Returns (test_acc, elapsed)."""
    return train_on_heterodata(data, target_type, device,
                               epochs=train_epochs, hidden=train_hidden)


def run_once(data, target_type, ratio, device, pretrain,
             train_epochs=200, train_hidden=256, verbose=False):
    """Run one full compress → train cycle.

    Returns a dict with compression ratios, timing, and test accuracy.
    """
    # ── Compression ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = hcgc.compress(
            data,
            ratio       = ratio,
            target_type = target_type,
            pretrain    = pretrain,
            device      = device,
            verbose     = verbose,
        )
    t_compress = time.perf_counter() - t0

    # ── Downstream training ───────────────────────────────────────────────────
    test_acc, t_train = train_downstream(
        result, target_type,
        device_str   = device,
        epochs       = train_epochs,
        hidden       = train_hidden,
    )

    n_orig = result.info['n_nodes_orig']
    n_comp = result.info['n_nodes_comp']
    e_orig = result.info['edges_orig']
    e_comp = result.info['edges_comp']

    return {
        'node_ratio':   n_comp / max(n_orig, 1),           # retention: smaller = more compressed
        'edge_ratio':   e_comp / max(e_orig, 1),
        'compression':  result.info['compression'],        # n_orig / n_comp (e.g. 10.3x)
        't_compress':   t_compress,                        # hcgc.compress() wall time (pretrain + coarsen)
        't_coarsen':    result.info['coarsen_time'],       # C++ kernel only
        't_train':      t_train,
        't_total':      t_compress + t_train,
        'test_acc':     test_acc,
        'n_nodes_orig': n_orig,
        'n_nodes_comp': n_comp,
        'edges_orig':   e_orig,
        'edges_comp':   e_comp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(mean, std, fmt='.3f'):
    return f"{mean:{fmt}} ± {std:{fmt}}"


def main():
    parser = argparse.ArgumentParser(
        description='HCGC benchmark: node/edge compression and timing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset',  required=True, choices=list(LOADERS),
                        help='Dataset to benchmark')
    parser.add_argument('--ratio',    type=float, default=0.1,
                        help='Node retention ratio  (0.1 = keep 10%% = 10x target)')
    parser.add_argument('--runs',     type=int,   default=3,
                        help='Number of timed measurement runs')
    parser.add_argument('--warmup',   type=int,   default=1,
                        help='Warmup runs before measurement (excluded from stats)')
    parser.add_argument('--device',   default='auto',
                        help="Compute device: 'auto', 'cpu', or 'cuda'")
    parser.add_argument('--root',     default='data',
                        help='Dataset download root')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Skip GNN pretrain (faster, slightly lower quality)')
    parser.add_argument('--train-epochs', type=int, default=200,
                        help='Downstream training epochs per run')
    parser.add_argument('--train-hidden', type=int, default=256,
                        help='Hidden dim for downstream GNN')
    parser.add_argument('--baseline', action='store_true',
                        help='Also train on the original graph and print speedup comparison')
    args = parser.parse_args()

    pretrain = not args.no_pretrain

    # ── Load dataset ──────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'='*W}")
    print(f"  HCGC Benchmark")
    print(f"{'='*W}")
    print(f"  dataset  : {args.dataset}")
    print(f"  ratio    : {args.ratio}  ({1/args.ratio:.1f}x target compression)")
    print(f"  pretrain : {pretrain}")
    print(f"  device   : {args.device}")
    print(f"  runs     : {args.warmup} warmup  +  {args.runs} timed")
    print(f"{'='*W}\n")

    print(f"Loading {args.dataset} ...")
    data, target_type = LOADERS[args.dataset](args.root)

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"  node types : {list(data.node_types)}")
    print(f"  edge types : {len(data.edge_types)}")
    print(f"  total nodes: {n_nodes:,}   total edges: {n_edges:,}")
    print(f"  target type: {target_type!r}")

    # ── Baseline (original graph training) ───────────────────────────────────
    base_records = []
    if args.baseline:
        print(f"\nBaseline: training on original graph ({args.runs} runs) ...")
        for i in range(args.runs):
            print(f"  baseline run {i+1}/{args.runs} ... ", end='', flush=True)
            acc, t = run_baseline(data, target_type, args.device,
                                  train_epochs=args.train_epochs,
                                  train_hidden=args.train_hidden)
            base_records.append({'t_train': t, 'test_acc': acc})
            print(f"t={t:.1f}s  test_acc={acc:.4f}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    # The first call to hcgc_module (C++ via pybind11) and the first PyTorch
    # CUDA kernel launch both carry one-time JIT / driver init overhead.
    # Warmup runs flush this overhead before measurement begins.
    # We run with pretrain=False so warmup finishes quickly.
    if args.warmup > 0:
        print(f"\nWarmup  ({args.warmup} run(s), pretrain=False) ...")
        for i in range(args.warmup):
            t_wu = time.perf_counter()
            run_once(data, target_type,
                     ratio=args.ratio, device=args.device,
                     pretrain=False, verbose=False)
            print(f"  warmup {i+1}/{args.warmup}  ({time.perf_counter()-t_wu:.1f}s)")

    # ── Timed runs ────────────────────────────────────────────────────────────
    print(f"\nTimed runs ({args.runs}) ...")
    records = []
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs} ... ", end='', flush=True)
        r = run_once(
            data, target_type,
            ratio        = args.ratio,
            device       = args.device,
            pretrain     = pretrain,
            train_epochs = args.train_epochs,
            train_hidden = args.train_hidden,
            verbose      = False,
        )
        records.append(r)
        print(
            f"node_ratio={r['node_ratio']:.3f}  "
            f"edge_ratio={r['edge_ratio']:.3f}  "
            f"t_total={r['t_total']:.1f}s  "
            f"test_acc={r['test_acc']:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    def stat(key):
        vals = [r[key] for r in records]
        return float(np.mean(vals)), float(np.std(vals))

    r0 = records[0]

    print(f"\n{'='*W}")
    print(f"  RESULTS   dataset={args.dataset}  ratio={args.ratio}  ({args.runs} runs)")
    print(f"{'='*W}")
    print(f"  {'Nodes':<28}: {r0['n_nodes_orig']:>10,}  ->  {r0['n_nodes_comp']:>8,}")
    print(f"  {'Edges':<28}: {r0['edges_orig']:>10,}  ->  {r0['edges_comp']:>8,}")
    print()
    node_m, node_s = stat('node_ratio')
    edge_m, edge_s = stat('edge_ratio')
    comp_m, comp_s = stat('compression')
    print(f"  {'Node retention ratio':<28}: {_fmt(node_m, node_s)}   ({comp_m:.2f}x compression)")
    print(f"  {'Edge retention ratio':<28}: {_fmt(edge_m, edge_s)}")
    print()
    tc_m,  tc_s  = stat('t_compress')
    tco_m, tco_s = stat('t_coarsen')
    tt_m,  tt_s  = stat('t_train')
    tot_m, tot_s = stat('t_total')
    print(f"  {'Time  compress() total':<28}: {_fmt(tc_m,  tc_s,  '.1f')} s"
          f"  (coarsen kernel: {tco_m:.1f} ± {tco_s:.1f} s)")
    print(f"  {'Time  train on comp. graph':<28}: {_fmt(tt_m,  tt_s,  '.1f')} s")
    print(f"  {'Time  total':<28}: {_fmt(tot_m, tot_s, '.1f')} s")
    print()
    acc_m, acc_s = stat('test_acc')
    print(f"  {'Test accuracy':<28}: {_fmt(acc_m, acc_s, '.4f')}")

    # ── Baseline comparison ───────────────────────────────────────────────────
    if base_records:
        b_t_m  = float(np.mean([r['t_train']  for r in base_records]))
        b_t_s  = float(np.std ([r['t_train']  for r in base_records]))
        b_acc_m = float(np.mean([r['test_acc'] for r in base_records]))
        b_acc_s = float(np.std ([r['test_acc'] for r in base_records]))

        train_speedup = b_t_m / max(tt_m, 1e-6)
        total_speedup = b_t_m / max(tot_m, 1e-6)
        acc_drop      = b_acc_m - acc_m

        print()
        print(f"  {'─'*58}")
        print(f"  {'Baseline vs HCGC comparison':^58}")
        print(f"  {'─'*58}")
        print(f"  {'':28}  {'Baseline':>10}  {'HCGC':>10}")
        print(f"  {'Train time (comp. graph only)':<28}  {b_t_m:>9.1f}s  {tt_m:>9.1f}s"
              f"  ({train_speedup:.1f}x faster)")
        print(f"  {'Total time (incl. compress)':<28}  {'—':>10}  {tot_m:>9.1f}s"
              f"  ({total_speedup:.1f}x vs baseline train)")
        print(f"  {'Test accuracy':<28}  {b_acc_m:>10.4f}  {acc_m:>10.4f}"
              f"  ({acc_drop:+.4f} drop)")

    print(f"{'='*W}\n")


if __name__ == '__main__':
    main()
