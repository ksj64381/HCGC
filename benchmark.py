#!/usr/bin/env python
"""
benchmark.py -- Reproduce HCGC compression benchmarks.

Supported datasets: imdb, dblp, lastfm, ogbn-mag, aminer, acm

Timing note
-----------
The first call to the C++ kernel (hcgc_module) and the first PyTorch backward
pass both incur JIT / CUDA-warmup overhead that can dominate short runs.
This script excludes warmup runs from all reported measurements; see --warmup.

Usage
-----
    python benchmark.py --dataset imdb     --ratio 0.1
    python benchmark.py --dataset dblp     --ratio 0.1 --runs 5
    python benchmark.py --dataset lastfm   --ratio 0.1
    python benchmark.py --dataset ogbn-mag --ratio 0.1 --runs 1 --warmup 1
    python benchmark.py --dataset aminer   --ratio 0.1 --baseline
    python benchmark.py --dataset acm      --ratio 0.1 --baseline
"""

import argparse
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, MessagePassing

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    # Graceful fallback: plain iterable with no bar
    def _tqdm(it, **kw):
        return it

import hcgc
from hcgc._baselines import compress_ahugc_style, compress_random_type


_COMPRESSORS = {
    'hcgc': 'HCGC',
    'cgc_type': 'CGC-type adaptation',
    'random_type': 'Random type-isolated',
    'ahugc_style': 'AH-UGC-style hash',
}


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


def _ensure_split_masks(data, target_type, train_ratio=0.6, val_ratio=0.2,
                        seed=42):
    """Create deterministic train/val/test masks when a dataset lacks them."""
    nd = data[target_type]
    if (getattr(nd, 'train_mask', None) is not None
            and getattr(nd, 'val_mask', None) is not None
            and getattr(nd, 'test_mask', None) is not None):
        return data

    if getattr(nd, 'y', None) is None:
        raise RuntimeError(
            f"{target_type!r} has no labels; cannot create split masks.")

    y = nd.y
    if y.dim() > 1:
        y = y.squeeze()
        nd.y = y

    valid = (y >= 0).nonzero(as_tuple=True)[0]
    n_valid = int(valid.numel())
    if n_valid < 3:
        raise RuntimeError(
            f"{target_type!r} has too few labeled nodes ({n_valid}) "
            "to create train/val/test masks.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = valid[torch.randperm(n_valid, generator=gen)]

    n_train = max(1, int(round(train_ratio * n_valid)))
    n_val = max(1, int(round(val_ratio * n_valid)))
    if n_train + n_val >= n_valid:
        n_val = max(1, n_valid - n_train - 1)
    n_test = n_valid - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n_valid - n_val - n_test)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(nd.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(nd.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(nd.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    nd.train_mask = train_mask
    nd.val_mask = val_mask
    nd.test_mask = test_mask
    print(f"  [{target_type}] split masks absent - created "
          f"{n_train:,}/{n_val:,}/{n_test:,} train/val/test masks "
          f"(seed={seed})")
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
    data = _ensure_split_masks(data, 'user')
    return data, 'user'


def load_ogbn_mag(root):
    """Load OGBN-MAG and initialise features for all node types.

    OGB's PygNodePropPredDataset never returns PyG HeteroData directly.
    Depending on the installed OGB version it returns either:
      - Old dict format: raw.num_nodes_dict / raw.x_dict / raw.edge_index_dict
      - Newer format   : raw is already a HeteroData-like object
    This loader normalises both into a proper PyG HeteroData with:
      - paper: 128-dim bag-of-words features (L2-normalised)
      - author: mean of paper features via 'writes' edges
      - institution: mean of author features via 'affiliated_with' edges
      - field_of_study: mean of paper features via 'has_topic' edges
      - forward + reverse edges for bidirectional message passing
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        sys.exit("ogbn-mag requires the ogb package:  pip install ogb")

    from torch_geometric.data import HeteroData as _HeteroData

    # PyTorch >= 2.6 changed torch.load default to weights_only=True,
    # which breaks OGB's internal torch.load calls on cached .pt files.
    # Patch for the duration of this call only, then restore.
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
    try:
        dataset   = PygNodePropPredDataset(name='ogbn-mag', root=f'{root}/ogbn-mag')
        raw       = dataset[0]
        split_idx = dataset.get_idx_split()
    finally:
        _torch.load = _orig_load

    _NODE_TYPES = ['paper', 'author', 'institution', 'field_of_study']

    # ── Normalise to HeteroData regardless of OGB version ────────────────────
    # Old OGB (<1.3.2ish) returns a Data-like object with dict attributes;
    # newer versions may return a proper HeteroData.  Both cases handled here.
    is_dict_fmt = hasattr(raw, 'num_nodes_dict') and raw.num_nodes_dict is not None
    if is_dict_fmt:
        print("  [ogbn-mag] OGB dict format detected — converting to HeteroData ...")
        data = _HeteroData()
        for nt in _NODE_TYPES:
            data[nt].num_nodes = raw.num_nodes_dict[nt]
        if raw.x_dict is not None and 'paper' in raw.x_dict:
            data['paper'].x = raw.x_dict['paper']
        if raw.y_dict is not None and 'paper' in raw.y_dict:
            data['paper'].y = raw.y_dict['paper']
        for et, ei in raw.edge_index_dict.items():
            data[et].edge_index = ei
    else:
        # Already HeteroData (or HeteroData-compatible)
        data = raw
        print(f"  [ogbn-mag] node types: {list(data.node_types)}")

    if 'paper' not in [nt for nt in _NODE_TYPES if data[nt].num_nodes > 0
                       if hasattr(data[nt], 'num_nodes')]:
        # safety check — just rely on the presence of x
        pass
    if not hasattr(data['paper'], 'x') or data['paper'].x is None:
        raise RuntimeError(
            "ogbn-mag 'paper' nodes have no features after loading.\n"
            f"  Delete the cache and redownload:  rmdir /s {root}\\ogbn-mag"
        )

    # ── Semantic feature init for non-paper types ─────────────────────────────
    # paper: L2-normalise 128-dim bag-of-words
    pf = data['paper'].x.float()
    pf = torch.nn.functional.normalize(pf, p=2, dim=1)
    data['paper'].x = pf

    def _avg_scatter(dst_n, src_feat, src_idx, dst_idx):
        """Scatter-mean: average src_feat[src_idx] into rows dst_idx."""
        dim = src_feat.shape[1]
        out = torch.zeros(dst_n, dim)
        cnt = torch.zeros(dst_n, dtype=torch.float32)
        chunk = 1 << 20  # 1M edges per chunk to avoid OOM
        for start in range(0, len(src_idx), chunk):
            sc = src_idx[start:start + chunk]
            dc = dst_idx[start:start + chunk]
            out.scatter_add_(0, dc.unsqueeze(1).expand(-1, dim), src_feat[sc])
            cnt.scatter_add_(0, dc, torch.ones(len(sc)))
        return out / cnt.clamp(min=1).unsqueeze(1)

    # author ← mean of papers they wrote  (author, writes, paper)
    print("  [ogbn-mag] Computing author features ...")
    ei_wp = data[('author', 'writes', 'paper')].edge_index      # [2,E] row0=author row1=paper
    data['author'].x = _avg_scatter(data['author'].num_nodes, pf, ei_wp[1], ei_wp[0])

    # institution ← mean of affiliated authors
    print("  [ogbn-mag] Computing institution features ...")
    ei_ai = data[('author', 'affiliated_with', 'institution')].edge_index
    data['institution'].x = _avg_scatter(
        data['institution'].num_nodes, data['author'].x, ei_ai[0], ei_ai[1])

    # field_of_study ← mean of papers with that topic
    print("  [ogbn-mag] Computing field_of_study features ...")
    ei_pt = data[('paper', 'has_topic', 'field_of_study')].edge_index
    data['field_of_study'].x = _avg_scatter(
        data['field_of_study'].num_nodes, pf, ei_pt[0], ei_pt[1])

    # ── Add reverse edges for bidirectional message passing ───────────────────
    data[('paper',         'rev_writes',         'author')].edge_index = ei_wp.flip(0).contiguous()
    data[('institution',   'rev_affiliated_with', 'author')].edge_index = ei_ai.flip(0).contiguous()
    data[('field_of_study','rev_has_topic',       'paper')].edge_index  = ei_pt.flip(0).contiguous()

    # ── Paper labels and split masks ──────────────────────────────────────────
    if data['paper'].y.dim() == 2:
        data['paper'].y = data['paper'].y.squeeze(1)

    n_paper = data['paper'].num_nodes
    for split, attr in [('train', 'train_mask'), ('valid', 'val_mask'), ('test', 'test_mask')]:
        mask = torch.zeros(n_paper, dtype=torch.bool)
        mask[split_idx[split]['paper']] = True
        setattr(data['paper'], attr, mask)

    print(f"  [ogbn-mag] papers={n_paper:,}  "
          f"train={data['paper'].train_mask.sum():,}  "
          f"val={data['paper'].val_mask.sum():,}  "
          f"test={data['paper'].test_mask.sum():,}")
    print(f"  [ogbn-mag] edge types: {list(data.edge_types)}")

    return data, 'paper'


def load_aminer(root):
    """AMiner-small: labeled authors + their 1-hop paper/venue neighborhood.

    PyG's full AMiner has 4.9 M nodes (the complete AMiner academic
    knowledge base).  This loader extracts the benchmark-sized subset by
    keeping only the ~6 564 labeled author nodes and the paper / venue
    nodes reachable in one hop, giving a graph of roughly 15–25 k nodes —
    comparable to the small AMiner used in HAN / MAGNN papers.

    Node types : author (~6 564), paper (subset), venue (subset).
    Target     : author, 4-class research area.
    Splits     : 60 / 20 / 20 random split (seed=42) on labeled authors.
    """
    from torch_geometric.datasets import AMiner
    from torch_geometric.data import HeteroData

    full   = AMiner(root=f'{root}/AMiner')[0]
    target = 'author'

    y_full = full[target].y
    if y_full is None:
        raise RuntimeError("AMiner 'author' nodes have no .y labels.")

    # ── Step 1: keep only labeled authors (y >= 0) ────────────────────────
    labeled_mask = y_full >= 0
    labeled_ids  = labeled_mask.nonzero(as_tuple=True)[0]   # old → kept
    a_old2new    = torch.full((full[target].num_nodes,), -1, dtype=torch.long)
    a_old2new[labeled_ids] = torch.arange(len(labeled_ids))

    # ── Step 2: collect papers linked to labeled authors ──────────────────
    paper_keep = torch.zeros(full['paper'].num_nodes, dtype=torch.bool)
    for et in full.edge_types:
        src_t, _, dst_t = et
        ei = full[et].edge_index
        if src_t == 'author' and dst_t == 'paper':
            paper_keep[ei[1, labeled_mask[ei[0]]]] = True
        if src_t == 'paper' and dst_t == 'author':
            paper_keep[ei[0, labeled_mask[ei[1]]]] = True

    paper_ids   = paper_keep.nonzero(as_tuple=True)[0]
    p_old2new   = torch.full((full['paper'].num_nodes,), -1, dtype=torch.long)
    p_old2new[paper_ids] = torch.arange(len(paper_ids))

    # ── Step 3: collect venues linked to kept papers ──────────────────────
    venue_keep = torch.zeros(full['venue'].num_nodes, dtype=torch.bool)
    for et in full.edge_types:
        src_t, _, dst_t = et
        ei = full[et].edge_index
        if src_t == 'paper' and dst_t == 'venue':
            venue_keep[ei[1, paper_keep[ei[0]]]] = True
        if src_t == 'venue' and dst_t == 'paper':
            venue_keep[ei[0, paper_keep[ei[1]]]] = True

    venue_ids  = venue_keep.nonzero(as_tuple=True)[0]
    v_old2new  = torch.full((full['venue'].num_nodes,), -1, dtype=torch.long)
    v_old2new[venue_ids] = torch.arange(len(venue_ids))

    old2new = {'author': a_old2new, 'paper': p_old2new, 'venue': v_old2new}

    # ── Step 4: build filtered HeteroData ────────────────────────────────
    data = HeteroData()

    # Node features / labels
    for nt, ids in [('author', labeled_ids),
                    ('paper',  paper_ids),
                    ('venue',  venue_ids)]:
        if hasattr(full[nt], 'x') and full[nt].x is not None:
            data[nt].x = full[nt].x[ids]
        data[nt].num_nodes = len(ids)
    data['author'].y = y_full[labeled_ids]

    # Edges: remap and drop edges involving removed nodes
    for et in full.edge_types:
        src_t, rel, dst_t = et
        if src_t not in old2new or dst_t not in old2new:
            continue
        ei  = full[et].edge_index
        src_new = old2new[src_t][ei[0]]
        dst_new = old2new[dst_t][ei[1]]
        keep    = (src_new >= 0) & (dst_new >= 0)
        if keep.any():
            data[et].edge_index = torch.stack(
                [src_new[keep], dst_new[keep]])

    # ── Step 5: 60/20/20 split on labeled authors ─────────────────────────
    n = len(labeled_ids)
    torch.manual_seed(42)
    perm  = torch.randperm(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr = torch.zeros(n, dtype=torch.bool)
    va = torch.zeros(n, dtype=torch.bool)
    te = torch.zeros(n, dtype=torch.bool)
    tr[perm[:n_tr]]            = True
    va[perm[n_tr:n_tr + n_va]] = True
    te[perm[n_tr + n_va:]]     = True
    data[target].train_mask = tr
    data[target].val_mask   = va
    data[target].test_mask  = te

    return data, target


def load_acm(root):
    """ACM paper network with the standard HAN / MAGNN benchmark split.

    Downloaded once from the DGL mirror (ACM.zip → ACM.mat).
    All labels are visible — no y=-1 hidden-test convention.

    Node types : paper (3 025), author (5 912), conference.
    Target     : paper, 3-class research area
                 (Database / Wireless Comms / Data Mining).
    Split      : 600 train / 300 val / 2 125 test (standard ACM split).
    Requires   : scipy  (pip install scipy)

    Why not HGBDataset?
        HGB's 'acm' format sets y=-1 for test nodes (competition style).
        After HCGC supernode majority-voting, test supernodes inherit y=-1
        and are never predicted correctly → 0 % test accuracy.
    """
    import os, urllib.request, zipfile
    try:
        import scipy.io, scipy.sparse
    except ImportError:
        sys.exit("ACM loader requires scipy:  pip install scipy")
    from torch_geometric.data import HeteroData

    acm_root = os.path.join(root, 'ACM_dgl')
    mat_path = os.path.join(acm_root, 'ACM.mat')

    if not os.path.exists(mat_path):
        os.makedirs(acm_root, exist_ok=True)
        zip_path = os.path.join(acm_root, 'ACM.zip')

        _mirrors = [
            'https://data.dgl.ai/dataset/ACM.zip',
            'https://dgl-data.s3-accelerate.amazonaws.com/dataset/ACM.zip',
            'https://dgl-data.s3.us-west-2.amazonaws.com/dataset/ACM.zip',
        ]
        # Browser-like User-Agent avoids Cloudflare / CDN 403 blocks.
        _headers = {
            'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64; rv:115.0) '
                           'Gecko/20100101 Firefox/115.0')
        }
        _downloaded = False
        for _url in _mirrors:
            try:
                print(f'  Downloading ACM from {_url} ...')
                req = urllib.request.Request(_url, headers=_headers)
                with urllib.request.urlopen(req, timeout=120) as _resp, \
                     open(zip_path, 'wb') as _f:
                    _f.write(_resp.read())
                _downloaded = True
                break
            except Exception as _e:
                print(f'  Failed ({type(_e).__name__}: {_e}), trying next ...')

        if not _downloaded:
            sys.exit(
                "\nAll ACM download mirrors failed.\n\n"
                "Manual download (choose one):\n"
                "  https://data.dgl.ai/dataset/ACM.zip\n"
                "  https://dgl-data.s3-accelerate.amazonaws.com/dataset/ACM.zip\n\n"
                "Then place ACM.mat at:\n"
                f"  {mat_path}\n"
                "(Unzip ACM.zip and move ACM.mat if needed.)"
            )

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(acm_root)
        os.remove(zip_path)
        # The zip may extract into a sub-folder; move mat file up if needed.
        nested = os.path.join(acm_root, 'ACM', 'ACM.mat')
        if os.path.exists(nested) and not os.path.exists(mat_path):
            import shutil
            shutil.move(nested, mat_path)

    mat = scipy.io.loadmat(mat_path)

    def _dense(m):
        return m.toarray() if scipy.sparse.issparse(m) else np.array(m)

    def _sp_to_ei(sp):
        if not scipy.sparse.issparse(sp):
            sp = scipy.sparse.coo_matrix(np.array(sp))
        coo = sp.tocoo()
        return torch.stack([
            torch.from_numpy(coo.row.astype('int64')),
            torch.from_numpy(coo.col.astype('int64')),
        ])

    # ── Paper features & labels ───────────────────────────────────────────────
    feat    = torch.from_numpy(_dense(mat['feature'])).float()
    lbl_oh  = _dense(mat['label'])
    labels  = torch.from_numpy(lbl_oh.argmax(axis=1).astype('int64'))
    n_paper = feat.shape[0]

    # ── Train / val / test masks ──────────────────────────────────────────────
    def _to_mask(key):
        idx = mat[key].flatten().astype('int64')
        if len(idx) > 0 and idx.max() >= n_paper:   # 1-based MATLAB indexing
            idx = idx - 1
        m = torch.zeros(n_paper, dtype=torch.bool)
        m[torch.from_numpy(idx)] = True
        return m

    data = HeteroData()
    data['paper'].x          = feat
    data['paper'].y          = labels
    data['paper'].train_mask = _to_mask('train_idx')
    data['paper'].val_mask   = _to_mask('val_idx')
    data['paper'].test_mask  = _to_mask('test_idx')

    # ── Paper–Author edges ────────────────────────────────────────────────────
    PA = mat['PvsA']
    data[('paper', 'written_by', 'author')].edge_index = _sp_to_ei(PA)
    data[('author', 'writes', 'paper')].edge_index     = _sp_to_ei(PA).flip(0)
    data['author'].num_nodes = int(PA.shape[1])

    # ── Paper–Paper citation edges ────────────────────────────────────────────
    if 'PvsP' in mat:
        data[('paper', 'cites', 'paper')].edge_index = _sp_to_ei(mat['PvsP'])

    # ── Paper–Conference edges (skip PvsL — it encodes the class labels) ──────
    for ck, nname in [('PvsC', 'conference'), ('PvsV', 'venue')]:
        if ck in mat and hasattr(mat[ck], 'shape') and mat[ck].shape[1] > 3:
            data[('paper', 'in', nname)].edge_index       = _sp_to_ei(mat[ck])
            data[(nname, 'contains', 'paper')].edge_index = _sp_to_ei(mat[ck]).flip(0)
            data[nname].num_nodes = int(mat[ck].shape[1])
            break

    return data, 'paper'


def load_freebase(root):
    """Load HGBn-Freebase: ~180k nodes, 8 types, 4 classes on book nodes.

    Freebase is a large knowledge graph with no precomputed node features.
    _add_degree_features() (called in main) injects a log-degree feature for
    every node type so the pipeline never sees a missing .x.

    Dataset stats (from the HGB benchmark, Lv et al. 2021):
      node types : 8  (book, film, music, sports, people,
                       location, organization, business)
      total nodes: ~180k
      target type: book  (4 classes)
      edges      : ~1M (much sparser than ogbn-mag → fast training)

    The HGB split is semi-supervised: few labeled training nodes, large test.
    Because train_ratio is low the inherited val_mask from build_compressed_data
    is NOT empty (unlike ogbn-mag), so _resplit_supernodes usually won't trigger.
    """
    try:
        from torch_geometric.datasets import HGBDataset
    except ImportError:
        sys.exit(
            "Freebase requires PyG >= 2.1 with HGBDataset.\n"
            "  pip install torch_geometric"
        )

    print("  [Freebase] Loading via HGBDataset (downloads on first run) ...")
    dataset = HGBDataset(root=f'{root}/Freebase', name='Freebase')
    data    = dataset[0]

    # ── Auto-detect target node type (first type with y + train_mask) ─────────
    target_type = None
    for nt in data.node_types:
        nd = data[nt]
        if (getattr(nd, 'y', None) is not None
                and getattr(nd, 'train_mask', None) is not None):
            target_type = nt
            break

    if target_type is None:
        # Fall back: any type with y
        for nt in data.node_types:
            if getattr(data[nt], 'y', None) is not None:
                target_type = nt
                break

    if target_type is None:
        raise RuntimeError(
            f"Could not detect target node type in Freebase.\n"
            f"  Available node types: {data.node_types}"
        )

    # ── Normalise y ───────────────────────────────────────────────────────────
    nd = data[target_type]
    if nd.y.dim() == 2:
        data[target_type].y = nd.y.squeeze(1)
    y = data[target_type].y

    # ── Build masks — handle all PyG / HGB format variants ───────────────────
    # PyG version differences:
    #   New: bool tensors  data['book'].train_mask / val_mask / test_mask
    #   Old: index tensors data['book'].train_idx  / val_idx  / test_idx
    # HGB Freebase quirk: val split is not in the raw files — PyG only produces
    #   train_mask + test_mask.  We carve out 20% of train as validation.
    nd = data[target_type]

    # Helper: idx tensor → bool mask
    def _idx_to_mask(idx_attr):
        idx = getattr(nd, idx_attr, None)
        if idx is None:
            return None
        m = torch.zeros(nd.num_nodes, dtype=torch.bool)
        m[idx] = True
        return m

    # train_mask
    if getattr(nd, 'train_mask', None) is None:
        m = _idx_to_mask('train_idx')
        if m is None:
            raise RuntimeError(f"Freebase: train_mask/train_idx not found "
                               f"for node type {target_type!r}.\n"
                               f"  Available keys: {list(nd.keys())}")
        data[target_type].train_mask = m

    # test_mask
    if getattr(nd, 'test_mask', None) is None:
        m = _idx_to_mask('test_idx')
        if m is None:
            raise RuntimeError(f"Freebase: test_mask/test_idx not found "
                               f"for node type {target_type!r}.\n"
                               f"  Available keys: {list(nd.keys())}")
        data[target_type].test_mask = m

    # val_mask — HGB Freebase raw format has no explicit val split
    # → carve out 20% of training nodes (reproducible with seed=0)
    nd = data[target_type]   # re-fetch
    if getattr(nd, 'val_mask', None) is None:
        m = _idx_to_mask('val_idx')
        if m is not None:
            data[target_type].val_mask = m
        else:
            # Create val by splitting train 80/20
            train_idx = nd.train_mask.nonzero(as_tuple=True)[0]
            gen = torch.Generator(); gen.manual_seed(0)
            perm    = torch.randperm(len(train_idx), generator=gen)
            n_val   = max(1, int(0.2 * len(train_idx)))
            val_nodes   = train_idx[perm[:n_val]]
            train_nodes = train_idx[perm[n_val:]]

            val_m   = torch.zeros(nd.num_nodes, dtype=torch.bool)
            train_m = torch.zeros(nd.num_nodes, dtype=torch.bool)
            val_m[val_nodes]   = True
            train_m[train_nodes] = True

            data[target_type].val_mask   = val_m
            data[target_type].train_mask = train_m
            print(f"  [Freebase] val_mask absent in raw data — "
                  f"carved {n_val:,} nodes from train as val (seed=0)")

    nd = data[target_type]   # re-fetch after possible setattr
    n_cls = int(y.max().item()) + 1
    n_tot = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)

    print(f"  [Freebase] target={target_type!r}  "
          f"nodes={nd.num_nodes:,}  classes={n_cls}")
    print(f"  [Freebase] train={nd.train_mask.sum():,}  "
          f"val={nd.val_mask.sum():,}  "
          f"test={nd.test_mask.sum():,}")
    print(f"  [Freebase] total nodes={n_tot:,}  "
          f"edge types={len(data.edge_types)}  "
          f"total edges={n_edges:,}")

    return data, target_type


LOADERS = {
    'imdb':      load_imdb,
    'dblp':      load_dblp,
    'lastfm':    load_lastfm,
    'ogbn-mag':  load_ogbn_mag,
    'aminer':    load_aminer,
    'acm':       load_acm,
    'freebase':  load_freebase,
}


# ══════════════════════════════════════════════════════════════════════════════
# Downstream GNN (HeteroSAGE, self-contained — no internal hcgc imports)
# ══════════════════════════════════════════════════════════════════════════════

class _HeteroSAGE(torch.nn.Module):
    """GraphSAGE adapted for heterogeneous graphs via HeteroConv."""

    def __init__(self, edge_types, feat_dims, hidden, num_classes, dropout=0.5,
                 num_layers=2):
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
        self.convs = torch.nn.ModuleList([_conv() for _ in range(num_layers)])
        self.clf   = torch.nn.Linear(hidden, num_classes)
        self.drop  = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict, target_type):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items()
             if nt.replace('.', '_') in self.proj}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return self.clf(h[target_type])


class _WeightedSAGEConv(MessagePassing):
    """GraphSAGE-style weighted neighbour mean for quotient graphs."""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin_neigh = torch.nn.Linear(in_channels, out_channels)
        self.lin_root = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        if edge_weight is None:
            edge_weight = x_src.new_ones(edge_index.size(1))
        edge_weight = edge_weight.to(dtype=x_src.dtype, device=x_src.device)
        out = self.propagate(edge_index, x=x_src, edge_weight=edge_weight,
                             size=(x_src.size(0), x_dst.size(0)))
        dst = edge_index[1]
        denom = x_src.new_zeros(x_dst.size(0))
        denom.scatter_add_(0, dst, edge_weight)
        out = out / denom.clamp(min=1.0).unsqueeze(-1)
        return self.lin_neigh(out) + self.lin_root(x_dst)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class _WeightedHeteroSAGE(torch.nn.Module):
    """Heterogeneous GraphSAGE that consumes per-relation edge_weight."""

    def __init__(self, edge_types, feat_dims, hidden, num_classes, dropout=0.5,
                 num_layers=2):
        super().__init__()
        self.proj = torch.nn.ModuleDict({
            nt.replace('.', '_'): torch.nn.Linear(d, hidden)
            for nt, d in feat_dims.items()
        })

        def _conv():
            return HeteroConv(
                {et: _WeightedSAGEConv(hidden, hidden) for et in edge_types},
                aggr='mean')

        self.convs = torch.nn.ModuleList([_conv() for _ in range(num_layers)])
        self.clf   = torch.nn.Linear(hidden, num_classes)
        self.drop  = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict, target_type, edge_weight_dict=None):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items()
             if nt.replace('.', '_') in self.proj}
        edge_weight_dict = edge_weight_dict or {}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict, edge_weight_dict=edge_weight_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return self.clf(h[target_type])


class _DownstreamHGT(torch.nn.Module):
    """Heterogeneous Graph Transformer (Hu et al., WWW 2020).

    Uses type-dependent attention weights; handles any number of node/edge types
    without per-type weight explosion via a shared projection.
    """

    def __init__(self, data, hidden, num_classes, dropout=0.5,
                 num_layers=2, num_heads=2):
        from torch_geometric.nn import HGTConv
        super().__init__()
        self._ntypes = [nt for nt in data.node_types
                        if hasattr(data[nt], 'x') and data[nt].x is not None]
        self.proj = torch.nn.ModuleDict({
            nt.replace('.', '_'): torch.nn.Linear(data[nt].x.shape[1], hidden)
            for nt in self._ntypes
        })
        self.convs = torch.nn.ModuleList([
            HGTConv(hidden, hidden, data.metadata(), num_heads)
            for _ in range(num_layers)
        ])
        self.clf  = torch.nn.Linear(hidden, num_classes)
        self.drop = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict, target_type):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            h = {k: (F.relu(self.drop(h_new[k])) if (h_new.get(k) is not None
                     and i < self.num_layers - 1)
                     else (h_new[k] if h_new.get(k) is not None else h[k]))
                 for k in h}
        return self.clf(h[target_type])


class _DownstreamGAT(torch.nn.Module):
    """Heterogeneous GAT: GATConv per relation type, wrapped in HeteroConv.

    Uses concat=False (mean over heads) so the hidden dimension stays constant
    across all layers regardless of the number of attention heads.
    """

    def __init__(self, data, hidden, num_classes, dropout=0.5,
                 num_layers=2, num_heads=4):
        from torch_geometric.nn import GATConv
        super().__init__()
        self._ntypes = [nt for nt in data.node_types
                        if hasattr(data[nt], 'x') and data[nt].x is not None]
        self.proj = torch.nn.ModuleDict({
            nt.replace('.', '_'): torch.nn.Linear(data[nt].x.shape[1], hidden)
            for nt in self._ntypes
        })
        def _make():
            return HeteroConv(
                {et: GATConv(hidden, hidden, heads=num_heads,
                             concat=False, dropout=dropout, add_self_loops=False)
                 for et in data.edge_types},
                aggr='mean')
        self.convs = torch.nn.ModuleList([_make() for _ in range(num_layers)])
        self.clf  = torch.nn.Linear(hidden, num_classes)
        self.drop = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict, target_type):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return self.clf(h[target_type])


class _DownstreamRGCN(torch.nn.Module):
    """Relational GCN (Schlichtkrull et al., ESWC 2018).

    A separate GraphConv weight matrix per relation type, aggregated with sum.
    """

    def __init__(self, data, hidden, num_classes, dropout=0.5, num_layers=2):
        from torch_geometric.nn import GraphConv
        super().__init__()
        self._ntypes = [nt for nt in data.node_types
                        if hasattr(data[nt], 'x') and data[nt].x is not None]
        self.proj = torch.nn.ModuleDict({
            nt.replace('.', '_'): torch.nn.Linear(data[nt].x.shape[1], hidden)
            for nt in self._ntypes
        })
        def _make():
            return HeteroConv(
                {et: GraphConv(hidden, hidden) for et in data.edge_types},
                aggr='sum')
        self.convs = torch.nn.ModuleList([_make() for _ in range(num_layers)])
        self.clf  = torch.nn.Linear(hidden, num_classes)
        self.drop = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x_dict, edge_index_dict, target_type):
        h = {nt: F.relu(self.proj[nt.replace('.', '_')](x))
             for nt, x in x_dict.items() if nt in self._ntypes}
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict)
            if i < self.num_layers - 1:
                h = {k: F.relu(self.drop(v)) for k, v in h.items() if v is not None}
        return self.clf(h[target_type])


_DOWNSTREAM_MODELS = ('sage', 'hgt', 'gat', 'rgcn')


def _edge_weight_dict(data):
    weights = {}
    for et in data.edge_types:
        if hasattr(data[et], 'edge_weight'):
            weights[et] = data[et].edge_weight
    return weights or None


def _build_downstream_model(data, target_type, model_name, hidden, num_classes,
                             dropout=0.5, dev=None, use_edge_weights=False):
    """Factory: instantiate a downstream GNN by name and move to device."""
    model_name = model_name.lower()
    if model_name == 'hgt':
        m = _DownstreamHGT(data, hidden, num_classes, dropout)
    elif model_name == 'gat':
        m = _DownstreamGAT(data, hidden, num_classes, dropout)
    elif model_name == 'rgcn':
        m = _DownstreamRGCN(data, hidden, num_classes, dropout)
    else:   # 'sage' (default)
        feat_dims = {nt: data[nt].x.shape[1]
                     for nt in data.node_types
                     if hasattr(data[nt], 'x') and data[nt].x is not None}
        if use_edge_weights:
            m = _WeightedHeteroSAGE(
                data.edge_types, feat_dims, hidden, num_classes, dropout)
        else:
            m = _HeteroSAGE(data.edge_types, feat_dims, hidden, num_classes, dropout)
    return m if dev is None else m.to(dev)


# Total-node threshold above which full-batch GNN training is replaced
# by NeighborLoader mini-batch training to avoid GPU OOM.
_FULL_BATCH_NODE_LIMIT = 100_000


def _resplit_supernodes(data, target_type, orig_data=None, seed=0):
    """Re-split compressed-graph supernodes to match original train/val/test ratios.

    The inherited mask logic in build_compressed_data uses
        val_mask = has_val & ~has_train
    which becomes nearly empty when the train ratio is high (e.g. ogbn-mag:
    85% train → nearly every supernode contains a train node → val ≈ 0).
    Re-splitting by proportion ensures the downstream GNN has a real validation
    set for early stopping.  The final accuracy is always evaluated on *original*
    test nodes via node_map, so this does not affect the benchmark metric.
    """
    n_super = data[target_type].num_nodes

    if orig_data is not None:
        tr_ratio = orig_data[target_type].train_mask.float().mean().item()
        va_ratio = orig_data[target_type].val_mask.float().mean().item()
    else:
        tr_ratio, va_ratio = 0.6, 0.2   # fallback: 60/20/20

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_super, generator=gen)
    n_tr = int(tr_ratio * n_super)
    n_va = int(va_ratio * n_super)

    tr_m = torch.zeros(n_super, dtype=torch.bool)
    va_m = torch.zeros(n_super, dtype=torch.bool)
    te_m = torch.zeros(n_super, dtype=torch.bool)
    tr_m[perm[:n_tr]]           = True
    va_m[perm[n_tr:n_tr + n_va]] = True
    te_m[perm[n_tr + n_va:]]    = True

    print(f"  [re-split] val_mask was empty → re-split {n_super:,} supernodes: "
          f"train={n_tr:,} ({tr_ratio*100:.1f}%) / "
          f"val={n_va:,} ({va_ratio*100:.1f}%) / "
          f"test={n_super-n_tr-n_va:,} ({(1-tr_ratio-va_ratio)*100:.1f}%)")

    data[target_type].train_mask = tr_m
    data[target_type].val_mask   = va_m
    data[target_type].test_mask  = te_m

    # Show label stats so we can see if purity is good enough for learning
    if hasattr(data[target_type], 'y') and data[target_type].y is not None:
        y = data[target_type].y
        if y.dtype == torch.long:
            n_cls = int(y.max().item()) + 1
            counts = torch.bincount(y, minlength=n_cls).float()
            top1_pct = counts.max().item() / n_super * 100
            print(f"  [re-split] supernode label dist: {n_cls} classes, "
                  f"majority class={top1_pct:.1f}% of supernodes")


def _train_mini_batch_downstream(data, target_type, device_str,
                                  epochs=200, lr=1e-3, patience=None,
                                  hidden=256, batch_size=512,
                                  model_name='sage',
                                  orig_data=None, node_map=None,
                                  use_soft_labels=False,
                                  use_edge_weights=False):
    """Mini-batch HeteroSAGE training via NeighborLoader.

    Used automatically by train_on_heterodata when the graph has more than
    _FULL_BATCH_NODE_LIMIT total nodes (e.g. AMiner, ogbn-mag).

    patience=None (default): run ALL epochs without early stopping — safest
    for compressed graphs whose supernode val-labels are noisy majority votes.
    Pass an explicit integer to re-enable early stopping (measured in epochs).

    Samples 2-hop neighbourhoods with at most 10 neighbours per edge type
    per hop; this keeps each mini-batch subgraph tractable while still
    giving each seed node a meaningful receptive field.
    """
    from torch_geometric.loader import NeighborLoader

    # NeighborLoader needs torch-sparse or pyg-lib for the actual C++ sampling.
    # Check both packages and surface a clear message for each failure mode.
    _backend_ok   = False
    _backend_diag = {}
    for _pkg, _pip in [('torch_sparse', 'torch-sparse'), ('pyg_lib', 'pyg-lib')]:
        try:
            __import__(_pkg)
            _backend_ok = True
            break
        except ImportError:
            _backend_diag[_pip] = 'not installed'
        except Exception as _e:
            # Installed but C-extension failed to load (version mismatch, missing .so, …)
            _backend_diag[_pip] = f'installed but failed to import ({type(_e).__name__}: {_e})'

    if not _backend_ok:
        _diag_lines = '\n'.join(
            f'  {pkg}: {msg}' for pkg, msg in _backend_diag.items()
        )
        raise RuntimeError(
            "\n\nNeighborLoader requires a *working* 'torch-sparse' or "
            "'pyg-lib' C-extension.\n\n"
            f"Diagnostic:\n{_diag_lines}\n\n"
            "Fix options:\n"
            "  1) Force-reinstall torch-sparse (most common fix):\n"
            "       pip install --force-reinstall torch-sparse\n\n"
            "  2) Use the version-specific wheel (replace X.Y.Z and cuZZZ):\n"
            "       python -c \"import torch; "
            "print(torch.__version__, torch.version.cuda)\"\n"
            "       pip install torch-sparse "
            "-f https://data.pyg.org/whl/torch-X.Y.Z+cuZZZ.html\n\n"
            "  3) Diagnose the raw import error:\n"
            "       python -c \"import torch_sparse\"\n"
        )

    dev = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_str == 'auto' else device_str)

    num_classes = int(data[target_type].y.max().item()) + 1
    model = _build_downstream_model(data, target_type, model_name,
                                    hidden, num_classes, dev=dev,
                                    use_edge_weights=use_edge_weights)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Per-edge-type neighbor counts: {et: [hop1, hop2]}.
    # Dense hetero graphs (e.g. ogbn-mag 7 edge types) blow up quickly with
    # large fan-outs; [5, 3] keeps subgraphs manageable while still capturing
    # 2-hop context.  Eval loaders use slightly larger batches but same counts.
    num_neighbors = {et: [5, 3] for et in data.edge_types}

    # num_workers=0: container environments often have very low fd limits
    # (EMFILE).  Multiple NeighborLoaders with workers>0 accumulate shared-memory
    # handles across train/val/test loaders + inference; keeping all loaders
    # synchronous eliminates the crash with negligible perf impact at these sizes.
    _n_workers = 0

    # Re-split supernodes proportionally when the inherited val_mask is empty.
    # build_compressed_data sets  val_mask = has_val & ~has_train,  which is
    # nearly empty when train_ratio is high (ogbn-mag: 85% train).
    # Must happen BEFORE loader creation so loaders see the updated masks.
    _n_val_pre = data[target_type].val_mask.sum().item()
    if _n_val_pre < 5 and orig_data is not None:
        _resplit_supernodes(data, target_type, orig_data)

    soft_y = None
    if use_soft_labels and hasattr(data[target_type], 'soft_y'):
        soft_y = data[target_type].soft_y
        print("  [train] Soft-label cross-entropy ON (mini-batch)")

    train_loader = NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size,
        input_nodes   = (target_type, data[target_type].train_mask),
        shuffle       = True,
        num_workers   = _n_workers,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size * 4,
        input_nodes   = (target_type, data[target_type].val_mask),
        shuffle       = False,
        num_workers   = _n_workers,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size * 4,
        input_nodes   = (target_type, data[target_type].test_mask),
        shuffle       = False,
        num_workers   = _n_workers,
    )

    def _eval(loader):
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(dev)
                out   = model(batch.x_dict, batch.edge_index_dict, target_type,
                              _edge_weight_dict(batch)) if use_edge_weights else \
                        model(batch.x_dict, batch.edge_index_dict, target_type)
                n     = batch[target_type].batch_size
                pred  = out[:n].argmax(dim=1)
                lbl   = batch[target_type].y[:n]
                valid = lbl >= 0          # exclude y=-1 (unlabeled supernodes)
                correct += (pred[valid] == lbl[valid]).sum().item()
                total   += valid.sum().item()
        return correct / max(total, 1)

    best_val, best_test = 0.0, 0.0
    best_state          = None
    no_improve          = 0
    eval_every          = 10

    # patience=None → run ALL epochs (patience_steps = epochs // eval_every means
    # we'd need no improvement for the ENTIRE training to early-stop, which never
    # happens within the epoch budget).  This is the safe default for compressed
    # graphs whose supernode labels are noisy majority votes.
    if patience is None:
        patience_steps = epochs // eval_every  # effectively "never stop early"
    else:
        patience_steps = max(patience // eval_every, 1)

    # Safety fallback: if val_mask is STILL empty after the resplit attempt above
    # (e.g. orig_data was None), disable early stopping rather than crash.
    n_val = data[target_type].val_mask.sum().item()
    if n_val < 5:
        print(f"  [WARNING] Only {int(n_val)} val supernodes even after resplit attempt — "
              f"disabling early stopping, training full {epochs} epochs")
        patience_steps = epochs // eval_every + 1

    t0 = time.perf_counter()

    ep_bar = _tqdm(range(1, epochs + 1), desc='  train', unit='ep',
                   ncols=88, leave=True)
    last_loss = float('nan')
    for ep in ep_bar:
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            batch = batch.to(dev)
            out   = model(batch.x_dict, batch.edge_index_dict, target_type,
                          _edge_weight_dict(batch)) if use_edge_weights else \
                    model(batch.x_dict, batch.edge_index_dict, target_type)
            n     = batch[target_type].batch_size
            if soft_y is not None:
                b_nids = batch[target_type].n_id[:n].cpu()
                b_soft = soft_y[b_nids].to(dev)
                loss = -(b_soft * F.log_softmax(out[:n], dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(out[:n], batch[target_type].y[:n],
                                       ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        sched.step()
        last_loss = total_loss / max(n_batches, 1)

        if ep % eval_every == 0:
            model.eval()
            val_acc  = _eval(val_loader)
            test_acc = _eval(test_loader)

            if val_acc > best_val:
                best_val, best_test, no_improve = val_acc, test_acc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            ep_bar.set_postfix(
                loss=f'{last_loss:.4f}',
                val=f'{val_acc:.4f}',
                best=f'{best_val:.4f}',
                pat=f'{no_improve}/{patience_steps}',
            )

            if no_improve >= patience_steps:
                ep_bar.set_description('  train [early-stop]')
                break
        else:
            ep_bar.set_postfix(loss=f'{last_loss:.4f}', best=f'{best_val:.4f}')

    elapsed = time.perf_counter() - t0

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    # -- Original-node evaluation (maps supernode predictions -> original labels) --
    if orig_data is not None and node_map is not None and target_type in node_map:
        from torch_geometric.loader import NeighborLoader
        n_target = data[target_type].num_nodes
        # Use dict format for hetero graphs so each edge type gets the same
        # per-type neighbor count instead of a confusing flat list.
        _inf_neighbors = {et: [10, 10] for et in data.edge_types}
        inf_loader = NeighborLoader(
            data,
            num_neighbors   = _inf_neighbors,
            batch_size      = batch_size * 4,
            input_nodes     = (target_type, torch.ones(n_target, dtype=torch.bool)),
            shuffle         = False,
            num_workers     = 0,
        )
        preds = torch.empty(n_target, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            for batch in inf_loader:
                batch = batch.to(dev)
                out   = model(batch.x_dict, batch.edge_index_dict, target_type,
                              _edge_weight_dict(batch)) if use_edge_weights else \
                        model(batch.x_dict, batch.edge_index_dict, target_type)
                bs    = batch[target_type].batch_size
                pred  = out[:bs].argmax(1).cpu()
                nids  = batch[target_type].n_id[:bs].cpu()
                preds[nids] = pred
        nm        = node_map[target_type]            # [n_orig] -> supernode idx
        te_mask   = orig_data[target_type].test_mask
        y_orig    = orig_data[target_type].y[te_mask]
        super_idx = nm[te_mask]                      # which supernode each orig test node belongs to
        orig_acc  = (preds[super_idx] == y_orig).float().mean().item()
        return orig_acc, elapsed

    return best_test, elapsed


def train_on_heterodata(data, target_type, device_str,
                        epochs=200, lr=1e-3, patience=None, hidden=256,
                        mini_batch_size=512, model_name='sage',
                        orig_data=None, node_map=None,
                        force_full_batch=False,
                        use_soft_labels=False,
                        use_edge_weights=False):
    """Train a 2-layer HeteroSAGE on any HeteroData object.

    Automatically switches to mini-batch (NeighborLoader) mode when the
    total node count exceeds _FULL_BATCH_NODE_LIMIT (100 k) to avoid OOM.
    Pass force_full_batch=True to skip this check (e.g. when the compressed
    graph fits in GPU memory even though the original does not).

    When orig_data and node_map are provided (compress() result), the final
    test accuracy is computed by mapping supernode predictions back to original
    test node labels — the same protocol used by HCGC's internal eval_pipeline.
    This gives a fairer accuracy number than evaluating on majority-vote
    supernode labels directly.

    Returns
    -------
    test_acc : float
    elapsed  : float   wall-clock seconds (model init + all epochs)
    """
    n_total = sum(data[nt].num_nodes for nt in data.node_types)
    if n_total > _FULL_BATCH_NODE_LIMIT and not force_full_batch:
        return _train_mini_batch_downstream(
            data, target_type, device_str,
            epochs=epochs, lr=lr, patience=patience, hidden=hidden,
            batch_size=mini_batch_size,
            model_name=model_name,
            orig_data=orig_data, node_map=node_map,
            use_soft_labels=use_soft_labels,
            use_edge_weights=use_edge_weights,
        )

    dev = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if device_str == 'auto' else device_str)

    # Re-split supernodes proportionally when the inherited val_mask is empty.
    # Must happen BEFORE clone+to(dev) so the GPU clone carries the new masks.
    _n_val_pre = data[target_type].val_mask.sum().item()
    if _n_val_pre < 5 and orig_data is not None:
        _resplit_supernodes(data, target_type, orig_data)

    cdata = data.clone().to(dev)  # clone: PyG .to() is in-place, original must stay on CPU

    num_classes = int(cdata[target_type].y.max().item()) + 1
    model = _build_downstream_model(cdata, target_type, model_name,
                                    hidden, num_classes, dev=dev,
                                    use_edge_weights=use_edge_weights)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    soft_y = None
    if use_soft_labels and hasattr(cdata[target_type], 'soft_y'):
        soft_y = cdata[target_type].soft_y
        print("  [train] Soft-label cross-entropy ON")

    best_val, best_test = 0.0, 0.0
    best_state          = None
    no_improve          = 0          # measured in eval steps, not epochs
    eval_every          = 10

    # patience=None → run ALL epochs (same logic as mini-batch path).
    if patience is None:
        patience_steps = epochs // eval_every   # effectively "never stop early"
    else:
        patience_steps = max(patience // eval_every, 1)

    # Safety fallback: if val_mask is STILL empty after the resplit attempt above
    # (e.g. orig_data was None), disable early stopping rather than crash.
    n_val = cdata[target_type].val_mask.sum().item()
    if n_val < 5:
        print(f"  [WARNING] Only {int(n_val)} val supernodes even after resplit attempt — "
              f"disabling early stopping, training full {epochs} epochs")
        patience_steps = epochs // eval_every + 1   # effectively never stop

    t0 = time.perf_counter()

    ep_bar = _tqdm(range(1, epochs + 1), desc='  train', unit='ep',
                   ncols=88, leave=True)
    edge_weight_dict = _edge_weight_dict(cdata) if use_edge_weights else None
    for ep in ep_bar:
        model.train()
        opt.zero_grad()
        out  = model(cdata.x_dict, cdata.edge_index_dict, target_type,
                     edge_weight_dict) if use_edge_weights else \
               model(cdata.x_dict, cdata.edge_index_dict, target_type)
        mask = cdata[target_type].train_mask
        if soft_y is not None:
            loss = -(soft_y[mask] * F.log_softmax(out[mask], dim=1)).sum(1).mean()
        else:
            loss = F.cross_entropy(out[mask], cdata[target_type].y[mask],
                                   ignore_index=-1)
        loss.backward()
        opt.step()
        sched.step()

        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                out = model(cdata.x_dict, cdata.edge_index_dict, target_type,
                            edge_weight_dict) if use_edge_weights else \
                      model(cdata.x_dict, cdata.edge_index_dict, target_type)
            pred = out.argmax(dim=1)
            y    = cdata[target_type].y

            def _masked_acc(mask):
                idx = mask & (y >= 0)   # exclude y=-1 unlabeled supernodes
                if idx.sum() == 0:
                    return 0.0
                return (pred[idx] == y[idx]).float().mean().item()

            if orig_data is not None and node_map is not None and target_type in node_map:
                nm = node_map[target_type].to(dev)

                def _orig_mapped_acc(mask_name):
                    o_mask = getattr(orig_data[target_type], mask_name)
                    if not o_mask.any():
                        return 0.0
                    o_mask_dev = o_mask.to(dev)
                    super_idx = nm[o_mask_dev]
                    y_orig = orig_data[target_type].y[o_mask].to(dev)
                    return (pred[super_idx] == y_orig).float().mean().item()

                val_acc  = _orig_mapped_acc('val_mask')
                test_acc = _orig_mapped_acc('test_mask')
            else:
                val_acc  = _masked_acc(cdata[target_type].val_mask)
                test_acc = _masked_acc(cdata[target_type].test_mask)

            if val_acc > best_val:
                best_val, best_test, no_improve = val_acc, test_acc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            ep_bar.set_postfix(
                loss=f'{loss.item():.4f}',
                val=f'{val_acc:.4f}',
                best=f'{best_val:.4f}',
                pat=f'{no_improve}/{patience_steps}',
            )

            if no_improve >= patience_steps:
                ep_bar.set_description('  train [early-stop]')
                break
        else:
            ep_bar.set_postfix(loss=f'{loss.item():.4f}', best=f'{best_val:.4f}')

    elapsed = time.perf_counter() - t0

    # Restore best-val checkpoint (fixes the bug where the final model state
    # is one early-stopping step past the best-val epoch)
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    # -- Original-node evaluation (maps supernode predictions -> original labels) --
    # When orig_data + node_map are supplied (compress() result), evaluate by
    # propagating supernode predictions back to original test node labels.
    # This is the same protocol as HCGC's internal eval_compressed_on_original().
    if orig_data is not None and node_map is not None and target_type in node_map:
        model.eval()
        with torch.no_grad():
            out = model(cdata.x_dict, cdata.edge_index_dict, target_type,
                        edge_weight_dict) if use_edge_weights else \
                  model(cdata.x_dict, cdata.edge_index_dict, target_type)
        nm        = node_map[target_type]            # [n_orig_target] -> supernode index
        te_mask   = orig_data[target_type].test_mask
        y_orig    = orig_data[target_type].y[te_mask]
        super_idx = nm[te_mask].to(dev)
        pred      = out[super_idx].argmax(1).cpu()
        orig_acc  = (pred == y_orig).float().mean().item()
        return orig_acc, elapsed

    return best_test, elapsed


def train_downstream(result, orig_data, target_type, device_str,
                     epochs=200, lr=1e-3, patience=None, hidden=256,
                     mini_batch_size=512, model_name='sage',
                     force_full_batch=False,
                     eval_protocol='original',
                     use_soft_labels=False,
                     use_edge_weights=False):
    """Train on a compressed HCGCResult.

    eval_protocol='original' maps predictions back to original test-node labels
    (the correct protocol for a coarsening benchmark). eval_protocol='supernode'
    reproduces the older majority-vote supernode-label evaluation.

    patience=None (default): run all epochs without early stopping. Compressed
    supernode labels are noisy majority votes, making early stopping unreliable.
    """
    eval_orig = orig_data if eval_protocol == 'original' else None
    eval_map = result.node_map if eval_protocol == 'original' else None
    return train_on_heterodata(result.data, target_type, device_str,
                               epochs=epochs, lr=lr, patience=patience,
                               hidden=hidden, mini_batch_size=mini_batch_size,
                               model_name=model_name,
                               orig_data=eval_orig, node_map=eval_map,
                               force_full_batch=force_full_batch,
                               use_soft_labels=use_soft_labels,
                               use_edge_weights=use_edge_weights)


def oracle_upper_bound(result, orig_data, target_type, mask_name='test_mask'):
    """Label-majority upper bound induced by the target-node compression map."""
    empty = {
        'oracle_acc': float('nan'),
        'oracle_n_nodes': 0,
        'oracle_n_supernodes': 0,
        'oracle_mixed_frac': float('nan'),
        'oracle_mean_purity': float('nan'),
    }
    if target_type not in result.node_map:
        return empty

    node_store = orig_data[target_type]
    mask = getattr(node_store, mask_name, None)
    if mask is None or not hasattr(node_store, 'y'):
        return empty

    y = node_store.y.detach().cpu()
    mask = mask.detach().cpu().bool()
    valid = mask & (y >= 0)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return empty

    super_idx = result.node_map[target_type].detach().cpu()[valid].long()
    labels = y[valid].long()
    n_cls = int(y[y >= 0].max().item()) + 1
    unique_super, inv = torch.unique(super_idx, return_inverse=True)

    counts = torch.zeros(len(unique_super), n_cls, dtype=torch.long)
    one_hot = torch.zeros(n_valid, n_cls, dtype=torch.long)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    counts.scatter_add_(0, inv.unsqueeze(1).expand(-1, n_cls), one_hot)

    per_super_total = counts.sum(dim=1).clamp(min=1)
    per_super_best = counts.max(dim=1).values
    classes_per_super = (counts > 0).sum(dim=1)

    return {
        'oracle_acc': float(per_super_best.sum().item() / n_valid),
        'oracle_n_nodes': n_valid,
        'oracle_n_supernodes': int(len(unique_super)),
        'oracle_mixed_frac': float((classes_per_super > 1).float().mean().item()),
        'oracle_mean_purity': float((per_super_best.float() / per_super_total.float()).mean().item()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single benchmark run
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(data, target_type, device, train_epochs=200, train_hidden=256,
                 mini_batch_size=512, model_name='sage', force_full_batch=False):
    """Train on the original (uncompressed) graph. Returns (test_acc, elapsed).

    Uses a generous patience (= train_epochs // 5) so the model trains until
    genuine convergence rather than stopping at an early local plateau.
    Large graphs (>100 k nodes) are handled automatically via mini-batch.
    """
    return train_on_heterodata(data, target_type, device,
                               epochs=train_epochs, hidden=train_hidden,
                               patience=train_epochs // 5,
                               mini_batch_size=mini_batch_size,
                               model_name=model_name,
                               force_full_batch=force_full_batch)


def run_once(data, target_type, ratio, device, pretrain,
             train_epochs=200, train_hidden=256, verbose=False,
             mini_batch_size=512, model_name='sage', force_full_batch=False,
             train_patience=None, emb_method='gnn',
             coarsen_l2_normalize=True, relprop_hops=2, relprop_outdim=128,
             pretrain_epochs=100, pretrain_patience=5,
             use_soft_labels=False, eval_protocol='original',
             pairwise_merge=False, type_thresholds=False,
             metapath_thresholds=False, edge_weight_mode='binary',
             freeze_node_types=None, compressor='hcgc',
             ratio_search='fast', auto_search_runs=8,
             auto_target_tolerance=None):
    """Run one full compress → train cycle.

    Returns a dict with compression ratios, timing, and test accuracy.
    train_patience=None (default): run all train_epochs, no early stopping.
    """
    # ── Compression ───────────────────────────────────────────────────────────
    compressor = str(compressor or 'hcgc').lower()
    if compressor not in _COMPRESSORS:
        raise ValueError(f"Unknown compressor={compressor!r}; "
                         f"choices={list(_COMPRESSORS)}")

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if compressor == 'random_type':
            result = compress_random_type(
                data,
                ratio=ratio,
                target_type=target_type,
                edge_weight_mode=edge_weight_mode,
                use_soft_labels=use_soft_labels,
                freeze_node_types=freeze_node_types,
                verbose=verbose,
            )
        elif compressor == 'ahugc_style':
            result = compress_ahugc_style(
                data,
                ratio=ratio,
                target_type=target_type,
                edge_weight_mode=edge_weight_mode,
                use_soft_labels=use_soft_labels,
                freeze_node_types=freeze_node_types,
                verbose=verbose,
            )
        else:
            result = hcgc.compress(
                data,
                ratio           = ratio,
                target_type     = target_type,
                pretrain        = pretrain,
                pretrain_epochs = pretrain_epochs,
                pretrain_patience = pretrain_patience,
                emb_method      = emb_method,
                coarsen_l2_normalize = coarsen_l2_normalize,
                relprop_hops    = relprop_hops,
                relprop_outdim  = relprop_outdim,
                device          = device,
                verbose         = verbose,
                mini_batch_size = mini_batch_size,
                use_soft_labels = use_soft_labels,
                pairwise_merge  = pairwise_merge or compressor == 'cgc_type',
                type_thresholds = False if compressor == 'cgc_type' else type_thresholds,
                metapath_thresholds = (
                    False if compressor == 'cgc_type' else metapath_thresholds),
                edge_weight_mode = edge_weight_mode,
                freeze_node_types = freeze_node_types,
                ratio_search = ratio_search,
                auto_search_runs = auto_search_runs,
                auto_target_tolerance = auto_target_tolerance,
            )
    t_compress = time.perf_counter() - t0
    oracle = oracle_upper_bound(result, data, target_type, 'test_mask')
    oracle_val = oracle_upper_bound(result, data, target_type, 'val_mask')

    # ── Downstream training ───────────────────────────────────────────────────
    # Default evaluation maps predictions back to original node labels; the
    # legacy supernode protocol is available through --eval-protocol supernode.
    test_acc, t_train = train_downstream(
        result, data, target_type,
        device_str       = device,
        epochs           = train_epochs,
        hidden           = train_hidden,
        mini_batch_size  = mini_batch_size,
        model_name       = model_name,
        force_full_batch = force_full_batch,
        patience         = train_patience,
        eval_protocol    = eval_protocol,
        use_soft_labels  = use_soft_labels,
        use_edge_weights = str(edge_weight_mode or 'binary').lower() != 'binary',
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
        'oracle_acc':   oracle['oracle_acc'],
        'oracle_val_acc': oracle_val['oracle_acc'],
        'oracle_gap':   oracle['oracle_acc'] - test_acc,
        'oracle_n_nodes': oracle['oracle_n_nodes'],
        'oracle_n_supernodes': oracle['oracle_n_supernodes'],
        'oracle_mixed_frac': oracle['oracle_mixed_frac'],
        'oracle_mean_purity': oracle['oracle_mean_purity'],
        'target_emb_distortion': result.info.get('target_emb_distortion', float('nan')),
        'target_emb_cosine': result.info.get('target_emb_cosine', float('nan')),
        'compressor': compressor,
        'ratio_search': ratio_search,
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
    parser.add_argument('--compressor', default='hcgc',
                        choices=list(_COMPRESSORS),
                        help='Compression method to evaluate. ahugc_style and '
                             'random_type are fast type-isolated baselines; '
                             'cgc_type is a CGC-like one-by-one adaptation.')
    parser.add_argument('--ratio-search', default='fast',
                        choices=['fast', 'precise'],
                        help='Target-ratio search mode for hcgc/cgc_type. '
                             'fast uses one-shot scale prediction; precise '
                             'uses bracket + log-space interpolation.')
    parser.add_argument('--auto-search-runs', type=int, default=8,
                        help='Max interpolation runs for --ratio-search precise.')
    parser.add_argument('--auto-target-tolerance', type=float, default=None,
                        help='Relative compression error tolerance. Default: '
                             '0.05 for precise, 0.15 for fast.')
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
    parser.add_argument('--emb-method', default='gnn',
                        choices=['gnn', 'fast', 'relprop', 'metapath2vec'],
                        help="Coarsening representation when pretrain is enabled. "
                             "'relprop' is training-free relation-aware propagation.")
    parser.add_argument('--raw-no-l2', action='store_true',
                        help='For --no-pretrain raw-feature coarsening, disable row-wise '
                             'L2 normalization before the C++ kernel.')
    parser.add_argument('--relprop-hops', type=int, default=2,
                        help='Propagation hops for --emb-method relprop')
    parser.add_argument('--relprop-outdim', type=int, default=128,
                        help='Output dimension for --emb-method relprop')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='Max epochs for GNN pretraining before coarsening')
    parser.add_argument('--pretrain-patience', type=int, default=5,
                        help='Early-stopping patience for GNN pretraining. '
                             'Eval is every 10 epochs, so values below 10 stop '
                             'after one non-improving eval.')
    parser.add_argument('--train-epochs', type=int, default=200,
                        help='Downstream training epochs per run')
    parser.add_argument('--train-hidden', type=int, default=256,
                        help='Hidden dim for downstream GNN')
    parser.add_argument('--mini-batch-size', type=int, default=512,
                        help='Seed-node batch size for downstream GNN training on large graphs '
                             '(>100 k nodes). Reduce if GPU OOM on dense graphs.')
    parser.add_argument('--force-full-batch', action='store_true',
                        help='Skip the 100k-node threshold and force full-batch downstream '
                             'training. Useful when the compressed graph fits in GPU memory '
                             'even if the original does not. May OOM on the baseline run.')
    parser.add_argument('--model',    default='sage', choices=list(_DOWNSTREAM_MODELS),
                        help='Downstream GNN architecture for evaluation')
    parser.add_argument('--train-patience', type=int, default=None,
                        help='Early-stopping patience in epochs for downstream training. '
                             'Default None = run all --train-epochs without early stopping '
                             '(recommended for compressed graphs with noisy supernode labels). '
                             'Use e.g. --train-patience 50 to re-enable early stopping.')
    parser.add_argument('--soft-labels', action='store_true',
                        help='Train compressed supernodes with class-proportion soft labels '
                             'instead of hard majority-vote labels.')
    parser.add_argument('--pairwise-merge', action='store_true',
                        help='CGC-like ablation: each density leader merges only the '
                             'single cheapest eligible neighbour under marginal join '
                             'cost instead of absorbing all neighbours inside the '
                             'Ball Multi-Merge radius.')
    parser.add_argument('--type-thresholds', action='store_true',
                        help='Estimate per-source-type threshold bases from '
                             'mediator-pair energy samples, then search one global '
                             'multiplier for target compression.')
    parser.add_argument('--metapath-thresholds', action='store_true',
                        help='Estimate per-(source type, mediator type) threshold '
                             'bases from mediator-pair energy samples. Takes '
                             'precedence over --type-thresholds.')
    parser.add_argument('--edge-weight-mode', default='binary',
                        choices=['binary', 'count', 'log_count', 'density'],
                        help='Compressed super-edge weighting. binary keeps the '
                             'current deduplicated quotient graph; count/log_count/'
                             'density preserve collapsed edge multiplicity and use '
                             'a weighted SAGE aggregator.')
    parser.add_argument('--freeze-node-types', nargs='*', default=[],
                        help='Node types to keep as singleton supernodes, e.g. '
                             '--freeze-node-types author for target-freeze ablation.')
    parser.add_argument('--eval-protocol', choices=['original', 'supernode'],
                        default='original',
                        help="'original' maps supernode predictions back to original test "
                             "nodes. 'supernode' reproduces the older optimistic benchmark "
                             "that evaluates majority-vote supernode labels.")
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
    print(f"  compressor: {args.compressor} ({_COMPRESSORS[args.compressor]})")
    print(f"  search   : {args.ratio_search}")
    print(f"  model    : {args.model}")
    print(f"  pretrain : {pretrain}")
    print(f"  emb      : {args.emb_method if pretrain else 'raw'}")
    if pretrain:
        print(f"  pretrain epochs/patience : {args.pretrain_epochs}/{args.pretrain_patience}")
    if not pretrain:
        print(f"  raw_l2   : {not args.raw_no_l2}")
    print(f"  soft_y   : {args.soft_labels}")
    print(f"  edge_w   : {args.edge_weight_mode}")
    print(f"  freeze   : {args.freeze_node_types or 'none'}")
    _merge_mode = 'pairwise' if args.pairwise_merge or args.compressor == 'cgc_type' else 'ball-multi'
    print(f"  merge    : {_merge_mode}")
    _thresh_mode = ('metapath-auto' if args.metapath_thresholds
                    else 'type-auto' if args.type_thresholds else 'global')
    print(f"  thresh   : {_thresh_mode}")
    print(f"  eval     : {args.eval_protocol}")
    print(f"  device   : {args.device}")
    print(f"  runs     : {args.warmup} warmup  +  {args.runs} timed")
    print(f"{'='*W}\n")

    print(f"Loading {args.dataset} ...")
    data, target_type = LOADERS[args.dataset](args.root)
    data = _add_degree_features(data)  # fill missing x for node types without features

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"  node types : {list(data.node_types)}")
    print(f"  edge types : {len(data.edge_types)}")
    print(f"  total nodes: {n_nodes:,}   total edges: {n_edges:,}")
    print(f"  target type: {target_type!r}")
    if n_nodes > _FULL_BATCH_NODE_LIMIT:
        if args.force_full_batch:
            print(f"  [large graph — --force-full-batch set: skipping mini-batch threshold]")
            print(f"  [WARNING] Baseline full-batch on {n_nodes:,} nodes may OOM on GPU]")
        else:
            print(f"  [large graph — downstream GNN will use mini-batch "
                  f"(batch_size={args.mini_batch_size})]")

    # ── Baseline (original graph training) ───────────────────────────────────
    base_records = []
    if args.baseline:
        print(f"\nBaseline: training on original graph ({args.runs} runs) ...")
        for i in range(args.runs):
            print(f"  baseline run {i+1}/{args.runs} ... ", end='', flush=True)
            acc, t = run_baseline(data, target_type, args.device,
                                  train_epochs=args.train_epochs,
                                  train_hidden=args.train_hidden,
                                  mini_batch_size=args.mini_batch_size,
                                  model_name=args.model,
                                  force_full_batch=args.force_full_batch)
            base_records.append({'t_train': t, 'test_acc': acc})
            print(f"t={t:.1f}s  test_acc={acc:.4f}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    # Two JIT sources need flushing before measurement:
    #   1. C++ kernel (hcgc_module): warmed up by pretrain=False run (fast)
    #   2. GNN pretrain code path:  warmed up by pretrain=True  run (same as timed)
    # Running both ensures no first-time compilation overhead leaks into results.
    if args.warmup > 0:
        print(f"\nWarmup  ({args.warmup} run(s)) ...")
        for i in range(args.warmup):
            # Pass 1: C++ kernel + basic PyTorch ops (fast, no pretrain)
            t_wu = time.perf_counter()
            run_once(data, target_type,
                     ratio=args.ratio, device=args.device,
                     pretrain=False, verbose=False,
                     mini_batch_size=args.mini_batch_size,
                     model_name=args.model,
                     force_full_batch=args.force_full_batch,
                     pretrain_epochs=args.pretrain_epochs,
                     pretrain_patience=args.pretrain_patience,
                     use_soft_labels=args.soft_labels,
                     pairwise_merge=args.pairwise_merge,
                     type_thresholds=args.type_thresholds,
                     metapath_thresholds=args.metapath_thresholds,
                     edge_weight_mode=args.edge_weight_mode,
                     eval_protocol=args.eval_protocol,
                     coarsen_l2_normalize=not args.raw_no_l2,
                     relprop_hops=args.relprop_hops,
                     relprop_outdim=args.relprop_outdim,
                     freeze_node_types=args.freeze_node_types,
                     compressor=args.compressor,
                     ratio_search=args.ratio_search,
                     auto_search_runs=args.auto_search_runs,
                     auto_target_tolerance=args.auto_target_tolerance)
            print(f"  warmup {i+1}/{args.warmup} [no-pretrain]  ({time.perf_counter()-t_wu:.1f}s)")
            # Pass 2: GNN pretrain code path (same config as timed runs)
            t_wu = time.perf_counter()
            run_once(data, target_type,
                     ratio=args.ratio, device=args.device,
                     pretrain=pretrain, verbose=False,
                     mini_batch_size=args.mini_batch_size,
                     model_name=args.model,
                     force_full_batch=args.force_full_batch,
                     emb_method=args.emb_method,
                     pretrain_epochs=args.pretrain_epochs,
                     pretrain_patience=args.pretrain_patience,
                     use_soft_labels=args.soft_labels,
                     pairwise_merge=args.pairwise_merge,
                     type_thresholds=args.type_thresholds,
                     metapath_thresholds=args.metapath_thresholds,
                     edge_weight_mode=args.edge_weight_mode,
                     eval_protocol=args.eval_protocol,
                     coarsen_l2_normalize=not args.raw_no_l2,
                     relprop_hops=args.relprop_hops,
                     relprop_outdim=args.relprop_outdim,
                     freeze_node_types=args.freeze_node_types,
                     compressor=args.compressor,
                     ratio_search=args.ratio_search,
                     auto_search_runs=args.auto_search_runs,
                     auto_target_tolerance=args.auto_target_tolerance)
            print(f"  warmup {i+1}/{args.warmup} "
                  f"[pretrain={pretrain}, emb={args.emb_method if pretrain else 'raw'}]  "
                  f"({time.perf_counter()-t_wu:.1f}s)")

    # ── Timed runs ────────────────────────────────────────────────────────────
    print(f"\nTimed runs ({args.runs}) ...")
    records = []
    for i in range(args.runs):
        print(f"  run {i+1}/{args.runs} ... ", end='', flush=True)
        r = run_once(
            data, target_type,
            ratio            = args.ratio,
            device           = args.device,
            pretrain         = pretrain,
            train_epochs     = args.train_epochs,
            train_hidden     = args.train_hidden,
            verbose          = False,
            mini_batch_size  = args.mini_batch_size,
            model_name       = args.model,
            force_full_batch = args.force_full_batch,
            train_patience   = args.train_patience,
            emb_method       = args.emb_method,
            pretrain_epochs  = args.pretrain_epochs,
            pretrain_patience = args.pretrain_patience,
            use_soft_labels  = args.soft_labels,
            pairwise_merge   = args.pairwise_merge,
            type_thresholds  = args.type_thresholds,
            metapath_thresholds = args.metapath_thresholds,
            edge_weight_mode = args.edge_weight_mode,
            eval_protocol    = args.eval_protocol,
            coarsen_l2_normalize = not args.raw_no_l2,
            relprop_hops     = args.relprop_hops,
            relprop_outdim   = args.relprop_outdim,
            freeze_node_types = args.freeze_node_types,
            compressor       = args.compressor,
            ratio_search     = args.ratio_search,
            auto_search_runs  = args.auto_search_runs,
            auto_target_tolerance = args.auto_target_tolerance,
        )
        records.append(r)
        print(
            f"node_ratio={r['node_ratio']:.3f}  "
            f"edge_ratio={r['edge_ratio']:.3f}  "
            f"t_total={r['t_total']:.1f}s  "
            f"test_acc={r['test_acc']:.4f}  "
            f"oracle={r['oracle_acc']:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    def stat(key):
        vals = np.array([r[key] for r in records], dtype=float)
        if np.isnan(vals).all():
            return float('nan'), float('nan')
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    r0 = records[0]

    node_m, node_s = stat('node_ratio')
    edge_m, edge_s = stat('edge_ratio')
    comp_m, comp_s = stat('compression')
    # Effective (edge-based) compression: GNN training cost scales with |E|,
    # so edge compression is the primary driver of training speedup.
    edge_comp_m = 1.0 / max(edge_m, 1e-9)
    edge_comp_s = edge_comp_m * (edge_s / max(edge_m, 1e-9))

    tc_m,  tc_s  = stat('t_compress')
    tco_m, tco_s = stat('t_coarsen')
    tt_m,  tt_s  = stat('t_train')
    tot_m, tot_s = stat('t_total')
    acc_m, acc_s = stat('test_acc')
    oracle_m, oracle_s = stat('oracle_acc')
    voracle_m, voracle_s = stat('oracle_val_acc')
    ogap_m, ogap_s = stat('oracle_gap')
    omix_m, omix_s = stat('oracle_mixed_frac')
    opur_m, opur_s = stat('oracle_mean_purity')
    ed_m, ed_s = stat('target_emb_distortion')
    ec_m, ec_s = stat('target_emb_cosine')

    comp_label = _COMPRESSORS[args.compressor]

    print(f"\n{'='*W}")
    print(f"  RESULTS   dataset={args.dataset}  compressor={args.compressor}  "
          f"target_ratio={args.ratio} ({1/args.ratio:.0f}x)  ({args.runs} runs)")
    print(f"{'='*W}")
    print(f"  {'Nodes':<28}: {r0['n_nodes_orig']:>10,}  ->  {r0['n_nodes_comp']:>8,}")
    print(f"  {'Edges':<28}: {r0['edges_orig']:>10,}  ->  {r0['edges_comp']:>8,}")
    print()
    print(f"  {'Node compression (actual)':<28}: {comp_m:.2f}x ± {comp_s:.2f}x"
          f"  (node ratio {node_m:.3f} ± {node_s:.3f})")
    print(f"  {'Edge compression (actual)':<28}: {edge_comp_m:.2f}x ± {edge_comp_s:.2f}x"
          f"  (edge ratio {edge_m:.3f} ± {edge_s:.3f})")
    print()
    print(f"  {'Time  compress() total':<28}: {_fmt(tc_m,  tc_s,  '.1f')} s"
          f"  (coarsen kernel: {tco_m:.1f} ± {tco_s:.1f} s)")
    print(f"  {'Time  train on comp. graph':<28}: {_fmt(tt_m,  tt_s,  '.1f')} s")
    print(f"  {'Time  total':<28}: {_fmt(tot_m, tot_s, '.1f')} s")
    print()
    print(f"  {'Test accuracy':<28}: {_fmt(acc_m, acc_s, '.4f')}")
    print(f"  {'Val oracle bound':<28}: {_fmt(voracle_m, voracle_s, '.4f')}")
    print(f"  {'Oracle upper bound':<28}: {_fmt(oracle_m, oracle_s, '.4f')}"
          f"  (gap {ogap_m:+.4f} 짹 {ogap_s:.4f})")
    print(f"  {'Oracle mixed supernodes':<28}: {_fmt(omix_m, omix_s, '.3f')}"
          f"  (mean purity {_fmt(opur_m, opur_s, '.3f')})")
    print(f"  {'Target emb distortion':<28}: {_fmt(ed_m, ed_s, '.4f')}"
          f"  (cosine {_fmt(ec_m, ec_s, '.4f')})")

    # ── Baseline comparison ───────────────────────────────────────────────────
    if base_records:
        b_t_m  = float(np.mean([r['t_train']  for r in base_records]))
        b_t_s  = float(np.std ([r['t_train']  for r in base_records]))
        b_acc_m = float(np.mean([r['test_acc'] for r in base_records]))
        b_acc_s = float(np.std ([r['test_acc'] for r in base_records]))

        train_speedup = b_t_m / max(tt_m, 1e-6)
        total_speedup = b_t_m / max(tot_m, 1e-6)
        acc_retention = acc_m / max(b_acc_m, 1e-9)   # fraction of baseline accuracy kept
        acc_drop      = acc_m - b_acc_m               # signed delta (negative = drop)

        print()
        print(f"  {'─'*58}")
        print(f"  {('Baseline vs ' + comp_label):^58}")
        print(f"  {'─'*58}")
        print(f"  {'':28}  {'Baseline':>10}  {args.compressor:>10}")
        print(f"  {'Train time':<28}  {b_t_m:>9.1f}s  {tt_m:>9.1f}s"
              f"  ({train_speedup:.1f}x faster)")
        print(f"  {'Total time (incl. compress)':<28}  {b_t_m:>9.1f}s  {tot_m:>9.1f}s"
              f"  ({total_speedup:.2f}x)")
        print(f"  {'Test accuracy':<28}  {b_acc_m:>10.4f}  {acc_m:>10.4f}"
              f"  ({acc_drop:+.4f},  {acc_retention*100:.1f}% retained)")

    print(f"{'='*W}\n")


if __name__ == '__main__':
    main()
