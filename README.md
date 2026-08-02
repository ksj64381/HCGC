# Type-Aware Heterogeneous Graph Coarsening via Mediator-Guided Coalition Formation

This repository implements **Heterogeneous Context-Guided Coarsening (HCGC)**,
a type-aware graph coarsening method for heterogeneous graphs. HCGC learns a
shared heterogeneous embedding space, forms same-type candidates through common
mediator nodes, applies type-aware pairwise merging, and projects the resulting
groups into a smaller `HeteroData` graph.

The repository contains source code only. Compiled Python extensions (`.pyd`,
`.so`, or `.dll`) are intentionally not tracked because they depend on the
operating system, Python ABI, and compiler toolchain.

## Installation

Use a Python environment with PyTorch and PyTorch Geometric. Install PyTorch for
the CUDA version available on the target machine, then install the remaining
dependencies:

```bash
python -m pip install -r requirements.txt
```

Build the C++ kernel locally before importing `hcgc`:

```bash
git clone https://github.com/ksj64381/HCGC.git
cd HCGC
python setup.py build_ext --inplace
```

The build creates `hcgc_module.*.pyd` on Windows or `hcgc_module.*.so` on
Linux/macOS in the project root. A C++14-compatible compiler and `pybind11` are
required. Rebuild the extension after changing Python versions or platforms.

## Quick Start

```python
from torch_geometric.datasets import IMDB
import hcgc

data = IMDB(root="data/IMDB")[0]
result = hcgc.compress(data, ratio=0.1)

compressed_data = result.data
actual_ratio = result.ratio
movie_to_supernode = result.node_map["movie"]
print(result.info)
```

`result.data` is a standard PyG `HeteroData` object. Predictions on compressed
target-type supernodes can be mapped back to original nodes with the
corresponding entry in `result.node_map`.

Selected API options:

```python
result = hcgc.compress(
    data,
    ratio=0.1,
    target_type=None,
    pretrain=True,
    pretrain_epochs=100,
    pretrain_patience=5,
    device="auto",
    pairwise_merge=True,
    edge_weight_mode="binary",
    ratio_search="fast",
)
```

The requested retention ratio is a target. The achieved ratio can differ
slightly because the number of retained groups is discrete; use `result.ratio`
and `result.info` when reporting compression.

## Datasets

`benchmark.py` provides loaders for IMDB, DBLP, ACM, and additional heterogeneous
graph datasets. PyG datasets are downloaded under `data/` by default. The ACM
experiments require `data/ACM_dgl/ACM.mat`.

The manuscript uses the conference-labeled three-class ACM variant. In
`paper_pipeline_ablation.py`, `--datasets acm --acm-variant paper` resolves to
the `acm3` loader. It is not the full ACM loader.

## Reproducing Manuscript Results

All commands below are run from the repository root after building the C++
extension. The manuscript configuration uses a maximum mediator candidate
budget of 128, binary projected edges, pairwise merging, hidden dimension 256,
and 200 downstream training epochs.

### Table I HCGC rows and Figure 3 HCGC curve

The final HCGC measurement uses ten timed seeds (`42` through `51`), one
discarded warm-up, and precise ratio search with a 2% relative retention
tolerance:

```bash
python -u paper_pipeline_ablation.py \
  --hcgc-root . \
  --datasets imdb dblp acm \
  --acm-variant paper \
  --compressors hcgc \
  --models sage rgcn gat appnp \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 \
  --max-candidates 128 \
  --emb-methods gnn \
  --runs 10 \
  --warmup 1 \
  --base-seed 42 \
  --device cuda \
  --ratio-search precise \
  --auto-search-runs 12 \
  --auto-target-tolerance 0.02 \
  --pairwise-merge \
  --edge-weight-mode binary \
  --pretrain-epochs 100 \
  --pretrain-patience 5 \
  --train-epochs 200 \
  --train-hidden 256 \
  --no-baseline \
  --plot-dir results/hcgc_k128_precise_10run
```

The runner writes run-level CSV/JSON records, checkpoint files, model-level and
cross-model summaries, plots, and `paper_pipeline_ablation_manifest.json`. The
manifest records the command, Git state, run seeds, library/CUDA versions, GPU
properties, ratio-search diagnostics, timings, memory measurements, storage,
and achieved compression.

### Table I controlled baselines

FreeHGC, AH-UGC, and CGC-Homo are controlled in-pipeline adaptations rather
than executions of the authors' official systems. Random-Type is a
type-constrained random-grouping baseline. All four share the same
compressed-graph construction and downstream evaluation interface:

```bash
python experiments.py \
  --datasets imdb dblp acm3 \
  --models sage rgcn gat appnp \
  --compressors random_type ahugc_style freehgc cgc_homo \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 \
  --runs 3 \
  --warmup 1 \
  --device cuda \
  --pairwise-merge \
  --edge-weight-mode binary \
  --train-epochs 200 \
  --train-hidden 256 \
  --plot-dir results/controlled_baselines
```

Do not describe these outputs as exact reproductions of FreeHGC, AH-UGC, or
the original CGC system. They preserve the methods' selection/grouping ideas
under the common evaluation pipeline used by this repository.

### Table II ablation study

The manuscript averages GraphSAGE, RGCN, GAT, and APPNP on IMDB, and GraphSAGE,
RGCN, and GAT on DBLP. Run the datasets separately so the model sets match the
reported table:

```bash
python ablation_experiments.py \
  --datasets imdb \
  --models sage rgcn gat appnp \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 \
  --runs 10 --warmup 1 \
  --ratio-search precise \
  --edge-weight-mode binary \
  --train-epochs 200 --train-hidden 256 \
  --device cuda \
  --plot-dir results/ablation_imdb_precise_10run

python ablation_experiments.py \
  --datasets dblp \
  --models sage rgcn gat \
  --ratios 0.5 0.3 0.25 0.2 0.15 0.1 \
  --runs 10 --warmup 1 \
  --ratio-search precise \
  --edge-weight-mode binary \
  --train-epochs 200 --train-hidden 256 \
  --device cuda \
  --plot-dir results/ablation_dblp_precise_10run
```

Warm-up executions are excluded from reported statistics. Report achieved
node-compression factors rather than assuming that every method exactly reaches
the requested target.

## Repository Layout

- `hcgc/`: public API, embedding, coarsening, graph construction, and baselines.
- `hcgc_module.cpp`: C++ coarsening kernel source.
- `setup.py`: local extension build configuration.
- `benchmark.py`: dataset loaders and single-configuration evaluation pipeline.
- `experiments.py`: multi-dataset/model/compressor sweeps.
- `paper_pipeline_ablation.py`: final HCGC measurement and provenance runner.
- `ablation_experiments.py`: core-component ablation runner.

Generated data, experiment outputs, build directories, and compiled extensions
are excluded from version control.

## Citation

The accompanying manuscript is:

> SeungJin Kim and Ikbeom Jang. "Type-Aware Heterogeneous Graph Coarsening via
> Mediator-Guided Coalition Formation."

HCGC adopts an iterative grouping motif from cooperative game-based graph
coarsening, but the method in this repository is formulated as context-guided
heterogeneous graph coarsening rather than as a new game-theoretic result.

The related CGC work is:

> Sonali Raj, Manoj Kumar, Sumit Kumar, Ruchir Gupta, and Amit Kumar Jaiswal.
> "Graph Coarsening using Game Theoretic Approach." TMLR, 2026.
> https://openreview.net/forum?id=5vLBjQJCln
