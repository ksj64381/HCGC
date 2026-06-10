# HCGC: Heterogeneous Graph Coarsening via Coalition Games

A 10× compressed graph typically trains 5–8× faster per run; accuracy retention depends on the dataset and compression ratio (see the paper for full results).

## Quick Start

```python
import hcgc

result = hcgc.compress(data, ratio=0.1)   # keep 10% of nodes = 10x compression

result.data      # compressed HeteroData — use directly for GNN training
result.ratio     # actual achieved ratio (e.g. 0.098 for ~10x)
result.node_map  # {node_type: LongTensor} original → supernode mapping
result.info      # {'compression': 10.2, 'n_nodes_orig': 1000000, ...}
```

## Installation

### Requirements

**Core library**

```bash
pip install torch torch-geometric numpy scikit-learn pybind11
```

**Benchmark datasets** (`benchmark.py` only — install what you need)

| Dataset | Extra package | Install |
|---------|--------------|---------|
| AMiner  | `pandas`     | `pip install pandas` |
| ogbn-mag | `ogb`       | `pip install ogb` |
| ACM (fallback) | `scipy` | `pip install scipy` |

Or install everything at once:

```bash
pip install -r requirements.txt
```

### How To Compile the C++ kernel

```bash
git clone https://github.com/ksj64381/HCGC.git
cd HCGC
python setup.py build_ext --inplace
```

After building, `hcgc_module.*.pyd` (Windows) or `hcgc_module.*.so` (Linux/macOS) will appear in the project root.


## API Reference

```python
hcgc.compress(
    data,                   # PyG HeteroData with node features + edge indices
    ratio        = 0.1,     # fraction of nodes to keep  (0.1 = 10x compression)
    target_type  = None,    # classification target node type (auto-detected)
    pretrain     = True,    # GNN pretrain → better embedding quality
    pretrain_epochs = 100,  # max pretrain epochs; early stopping applies
    device       = 'auto',  # 'auto' | 'cpu' | 'cuda'
    verbose      = True,
) -> HCGCResult
```

**`HCGCResult` fields:**

| Field | Type | Description |
|---|---|---|
| `.data` | `HeteroData` | Compressed graph, ready for PyG GNN training |
| `.ratio` | `float` | Actual node retention ratio achieved |
| `.node_map` | `dict` | `{node_type: LongTensor}` mapping original → supernode |
| `.info` | `dict` | Detailed stats: compression factor, node/edge counts, timing |

## Examples

### Basic compression

```python
from torch_geometric.datasets import IMDB
import hcgc

dataset = IMDB(root='/tmp/IMDB')
data = dataset[0]

result = hcgc.compress(data, ratio=0.1)
print(result.info)
# {'compression': 10.3, 'n_nodes_orig': 12772, 'n_nodes_comp': 1240,
#  'coarsen_time': 2.4, 'edges_orig': 37288, 'edges_comp': 3841, ...}
```

### Use compressed graph for GNN training

```python
from torch_geometric.nn import HeteroConv, SAGEConv
import torch

# result.data is a standard HeteroData — plug into any PyG model
compressed_data = result.data

# Map predictions back to original nodes
# result.node_map['movie'] is a LongTensor of shape [n_orig_movie]
# where result.node_map['movie'][i] is the supernode index for original node i
supernode_pred = model(compressed_data)           # shape: [n_supernodes]
original_pred  = supernode_pred[result.node_map['movie']]  # shape: [n_orig]
```

### No pretrain (faster, slightly lower quality)

```python
result = hcgc.compress(data, ratio=0.2, pretrain=False)
```

### Specify target type explicitly

```python
result = hcgc.compress(data, ratio=0.1, target_type='paper')
```

## Reproducing Paper Results

Main results (Tables I–II, Fig. 3):

```bash
python experiments.py --dataset imdb dblp acm3 \
  --models sage rgcn gat appnp \
  --compressors hcgc freehgc cgc_homo random_type ahugc_style \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 \
  --runs 3 --warmup 1 --device cuda --plot-dir sweep_results
```

Ablation study (Table III):

```bash
python ablation_experiments.py --datasets imdb dblp --models sage rgcn gat appnp \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 --runs 10 --warmup 1 \
  --ratio-search precise --device cuda --plot-dir ablation_results
```

## Rebuilding the C++ Kernel

If the pre-built binary doesn't match your environment:

```bash
pip install pybind11
python setup.py build_ext --inplace
```

Requirements: a C++14-compatible compiler (MSVC on Windows, GCC/Clang on Linux/macOS).

## Citation

This work extends ideas from:

> Sonali Raj, Manoj Kumar, Sumit Kumar, Ruchir Gupta, Amit Kumar Jaiswal.
> "Graph Coarsening using Game Theoretic Approach." TMLR, 2026.
> https://openreview.net/forum?id=5vLBjQJCln

HCGC extends the original CGC algorithm to heterogeneous graphs
with GNN-guided embedding and automatic compression control.
