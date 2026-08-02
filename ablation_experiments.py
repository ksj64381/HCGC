"""
Run HCGC ablation sweeps with the same evaluation path as experiments.py.

Default scope is intended for the paper ablation:

    datasets: imdb dblp
    models  : sage rgcn gat appnp
    ratios  : 0.5 0.3 0.25 0.2 0.15 0.1
    seeds   : 42 through 51
    candidate budget: 128

Examples
--------
    python ablation_experiments.py --runs 10 --warmup 1 --device cuda \
        --ratio-search precise --auto-search-runs 12 \
        --auto-target-tolerance 0.02 --max-candidates 128 \
        --no-baseline --plot-dir ablation_results

    # quick smoke test
    python ablation_experiments.py --datasets imdb --models sage \
        --variants full no_embedding --ratios 0.5 --runs 1 --warmup 0 \
        --train-epochs 1 --plot-dir ablation_smoke
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmark import LOADERS, _DOWNSTREAM_MODELS
from experiments import run_sweep


VARIANTS = {
    "full": {
        "label": "Full HCGC",
        "description": "Default HCGC: GNN embedding, pairwise Ward merge, reassignment.",
        "overrides": {},
    },
    "no_embedding": {
        "label": "w/o heterogeneous embedding",
        "description": "Disable GNN pretraining and coarsen on initialized/raw features.",
        "overrides": {"pretrain": False},
    },
    "fast_embedding": {
        "label": "fast embedding",
        "description": "Use training-free fast propagation embeddings instead of GNN pretraining.",
        "overrides": {"emb_method": "fast"},
    },
    "relprop_embedding": {
        "label": "relation propagation embedding",
        "description": "Use training-free relation-aware propagation embeddings.",
        "overrides": {"emb_method": "relprop"},
    },
    "ball_multi": {
        "label": "w/o pairwise admission",
        "description": "Use ball-style multi-merge instead of pairwise coalition formation.",
        "overrides": {"pairwise_merge": False},
    },
    "no_reassign": {
        "label": "w/o reassignment",
        "description": "Disable the post-merge reassignment heuristic.",
        "overrides": {"skip_reassignment": True},
    },
    "type_thresholds": {
        "label": "type thresholds",
        "description": "Use auto-calibrated per-source-type thresholds.",
        "overrides": {"type_thresholds": True},
    },
    "metapath_thresholds": {
        "label": "mediator thresholds",
        "description": "Use auto-calibrated per-(source type, mediator type) thresholds.",
        "overrides": {"metapath_thresholds": True},
    },
    "quotient_de": {
        "label": "quotient density objective",
        "description": "Use quotient density/embedding objective instead of Ward cost.",
        "overrides": {"merge_objective": "quotient_de"},
    },
    "raw_no_l2": {
        "label": "w/o embedding L2 norm",
        "description": "Disable L2 normalization for coarsening embeddings.",
        "overrides": {"coarsen_l2_normalize": False},
    },
    "edge_count": {
        "label": "count edge weights",
        "description": "Use projected edge multiplicity as coarsened edge weights.",
        "overrides": {"edge_weight_mode": "count"},
    },
    "edge_density": {
        "label": "density edge weights",
        "description": "Use density-normalized projected edge weights.",
        "overrides": {"edge_weight_mode": "density"},
    },
}

DEFAULT_VARIANTS = [
    "full",
    "no_embedding",
    "ball_multi",
    "no_reassign",
]


ROW_FIELDS = [
    "dataset",
    "model",
    "variant",
    "variant_label",
    "ratio",
    "runs",
    "base_seed",
    "timed_run_seeds",
    "max_candidates",
    "auto_search_runs",
    "auto_target_tolerance",
    "pretrain_epochs",
    "pretrain_patience",
    "compression",
    "node_compression",
    "node_ratio",
    "edge_compression",
    "edge_ratio",
    "storage_compression",
    "storage_ratio",
    "storage_reduction",
    "storage_orig_bytes",
    "storage_comp_bytes",
    "storage_comp_graph_bytes",
    "storage_map_bytes",
    "test_acc_mean",
    "test_acc_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "acc_drop",
    "macro_f1_drop",
    "val_oracle",
    "test_oracle",
    "emb_dist",
    "emb_cos",
    "t_total",
    "t_compress",
    "t_train",
    "train_speedup",
    "baseline_acc_mean",
    "baseline_acc_std",
    "baseline_macro_f1_mean",
    "baseline_macro_f1_std",
    "baseline_t_train",
    "ratio_search",
    "merge_objective",
    "edge_weight_mode",
    "pretrain",
    "emb_method",
    "pairwise_merge",
    "skip_reassignment",
    "type_thresholds",
    "metapath_thresholds",
    "variant_flags",
]

RUN_FIELDS = [
    "dataset",
    "model",
    "variant",
    "variant_label",
    "ratio",
    "run",
    "seed",
    "max_candidates",
    "compression",
    "node_ratio",
    "edge_compression",
    "edge_ratio",
    "storage_compression",
    "test_acc",
    "test_macro_f1",
    "test_micro_f1",
    "val_oracle",
    "test_oracle",
    "emb_dist",
    "emb_cos",
    "t_total",
    "t_compress",
    "t_train",
    "resplit_triggered",
    "variant_flags",
]

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "variant",
    "variant_label",
    "n_points",
    "mean_acc",
    "mean_drop",
    "mean_compression",
    "mean_val_oracle",
    "mean_emb_dist",
    "mean_t_total",
    "best_ratio",
    "best_compression",
    "best_acc",
]


def _json_clean(obj):
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


def _finite_mean(values):
    vals = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fv):
            vals.append(fv)
    return float(np.mean(vals)) if vals else float("nan")


def _format_float(v, digits=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(f):
        return "nan"
    return f"{f:.{digits}f}"


def _variant_config(name, args):
    spec = VARIANTS[name]
    cfg = {
        "pretrain": True,
        "emb_method": args.emb_method,
        "coarsen_l2_normalize": not args.raw_no_l2,
        "type_thresholds": False,
        "metapath_thresholds": False,
        "edge_weight_mode": args.edge_weight_mode,
        "pairwise_merge": True,
        "merge_objective": args.merge_objective,
        "skip_reassignment": False,
        "compressor": "hcgc",
    }
    cfg.update(spec["overrides"])
    return cfg


def _variant_flags(cfg):
    return {
        k: cfg[k]
        for k in [
            "pretrain",
            "emb_method",
            "coarsen_l2_normalize",
            "type_thresholds",
            "metapath_thresholds",
            "edge_weight_mode",
            "pairwise_merge",
            "merge_objective",
            "skip_reassignment",
            "compressor",
        ]
    }


def _row_from_entry(dataset, model, variant, cfg, base_stats, entry, args):
    base_acc = float("nan")
    base_std = float("nan")
    base_macro = float("nan")
    base_macro_std = float("nan")
    base_time = float("nan")
    if base_stats is not None:
        base_acc = base_stats.get("acc_mean", float("nan"))
        base_std = base_stats.get("acc_std", float("nan"))
        base_macro = base_stats.get("macro_f1_mean", float("nan"))
        base_macro_std = base_stats.get("macro_f1_std", float("nan"))
        base_time = base_stats.get("t_mean", float("nan"))

    flags = _variant_flags(cfg)
    return {
        "dataset": dataset,
        "model": model,
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "ratio": entry["ratio"],
        "runs": args.runs,
        "base_seed": args.base_seed,
        "timed_run_seeds": json.dumps(entry.get("run_seeds", [])),
        "max_candidates": args.max_candidates,
        "auto_search_runs": args.auto_search_runs,
        "auto_target_tolerance": args.auto_target_tolerance,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_patience": args.pretrain_patience,
        "compression": entry["comp_mean"],
        "node_compression": entry["comp_mean"],
        "node_ratio": entry.get("node_ratio_mean", float("nan")),
        "edge_compression": entry.get("edge_comp_mean", float("nan")),
        "edge_ratio": entry.get("edge_ratio_mean", float("nan")),
        "storage_compression": entry.get("storage_comp_mean", float("nan")),
        "storage_ratio": entry.get("storage_ratio_mean", float("nan")),
        "storage_reduction": entry.get("storage_reduction_mean", float("nan")),
        "storage_orig_bytes": entry.get("storage_orig_bytes_mean", float("nan")),
        "storage_comp_bytes": entry.get("storage_comp_bytes_mean", float("nan")),
        "storage_comp_graph_bytes": entry.get("storage_comp_graph_bytes_mean", float("nan")),
        "storage_map_bytes": entry.get("storage_map_bytes_mean", float("nan")),
        "test_acc_mean": entry["acc_mean"],
        "test_acc_std": entry["acc_std"],
        "test_macro_f1_mean": entry.get("macro_f1_mean", float("nan")),
        "test_macro_f1_std": entry.get("macro_f1_std", float("nan")),
        "acc_drop": entry.get("acc_drop", float("nan")),
        "macro_f1_drop": entry.get("macro_f1_drop", float("nan")),
        "val_oracle": entry.get("val_oracle_mean", float("nan")),
        "test_oracle": entry.get("oracle_mean", float("nan")),
        "emb_dist": entry.get("emb_dist_mean", float("nan")),
        "emb_cos": entry.get("emb_cos_mean", float("nan")),
        "t_total": entry.get("tt_mean", float("nan")),
        "t_compress": entry.get("tc_mean", float("nan")),
        "t_train": entry.get("tr_mean", float("nan")),
        "train_speedup": entry.get("train_speedup", float("nan")),
        "baseline_acc_mean": base_acc,
        "baseline_acc_std": base_std,
        "baseline_macro_f1_mean": base_macro,
        "baseline_macro_f1_std": base_macro_std,
        "baseline_t_train": base_time,
        "ratio_search": args.ratio_search,
        "merge_objective": cfg["merge_objective"],
        "edge_weight_mode": cfg["edge_weight_mode"],
        "pretrain": cfg["pretrain"],
        "emb_method": cfg["emb_method"],
        "pairwise_merge": cfg["pairwise_merge"],
        "skip_reassignment": cfg["skip_reassignment"],
        "type_thresholds": cfg["type_thresholds"],
        "metapath_thresholds": cfg["metapath_thresholds"],
        "variant_flags": json.dumps(flags, sort_keys=True),
    }


def _run_rows_from_entry(dataset, model, variant, cfg, entry, args):
    flags_json = json.dumps(_variant_flags(cfg), sort_keys=True)
    rows = []
    for run_idx, record in enumerate(entry.get("run_records", [])):
        rows.append({
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "variant_label": VARIANTS[variant]["label"],
            "ratio": entry["ratio"],
            "run": record.get("run", run_idx),
            "seed": record.get("seed", args.base_seed + run_idx),
            "max_candidates": record.get(
                "max_candidates", args.max_candidates),
            "compression": record.get("compression", float("nan")),
            "node_ratio": record.get("node_ratio", float("nan")),
            "edge_compression": record.get(
                "edge_compression", float("nan")),
            "edge_ratio": record.get("edge_ratio", float("nan")),
            "storage_compression": record.get(
                "storage_compression", float("nan")),
            "test_acc": record.get("test_acc", float("nan")),
            "test_macro_f1": record.get("test_macro_f1", float("nan")),
            "test_micro_f1": record.get("test_micro_f1", float("nan")),
            "val_oracle": record.get("oracle_val_acc", float("nan")),
            "test_oracle": record.get("oracle_acc", float("nan")),
            "emb_dist": record.get(
                "target_emb_distortion", float("nan")),
            "emb_cos": record.get("target_emb_cosine", float("nan")),
            "t_total": record.get("t_total", float("nan")),
            "t_compress": record.get("t_compress", float("nan")),
            "t_train": record.get("t_train", float("nan")),
            "resplit_triggered": record.get("resplit_triggered", False),
            "variant_flags": flags_json,
        })
    return rows


def _completed_keys(rows, ratios):
    required = {round(float(r), 8) for r in ratios}
    got = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["variant"])
        got.setdefault(key, set()).add(round(float(row["ratio"]), 8))
    return {key for key, vals in got.items() if required.issubset(vals)}


def _load_existing(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows):
    groups = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["variant"])
        groups.setdefault(key, []).append(row)

    out = []
    for (dataset, model, variant), items in sorted(groups.items()):
        best = max(items, key=lambda r: float(r["test_acc_mean"]))
        out.append({
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "variant_label": items[0]["variant_label"],
            "n_points": len(items),
            "mean_acc": _finite_mean(r["test_acc_mean"] for r in items),
            "mean_drop": _finite_mean(r["acc_drop"] for r in items),
            "mean_compression": _finite_mean(r["compression"] for r in items),
            "mean_val_oracle": _finite_mean(r["val_oracle"] for r in items),
            "mean_emb_dist": _finite_mean(r["emb_dist"] for r in items),
            "mean_t_total": _finite_mean(r["t_total"] for r in items),
            "best_ratio": best["ratio"],
            "best_compression": best["compression"],
            "best_acc": best["test_acc_mean"],
        })
    return out


def _write_markdown_table(path, summary_rows):
    lines = [
        "# HCGC Ablation Summary",
        "",
        "| Dataset | Model | Variant | Mean Acc | Mean Drop | Mean Comp. | Best Ratio | Best Acc |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {model} | {variant} | {mean_acc} | {mean_drop} | "
            "{mean_comp}x | {best_ratio} | {best_acc} |".format(
                dataset=row["dataset"],
                model=row["model"],
                variant=row["variant"],
                mean_acc=_format_float(row["mean_acc"]),
                mean_drop=_format_float(row["mean_drop"]),
                mean_comp=_format_float(row["mean_compression"], 2),
                best_ratio=_format_float(row["best_ratio"], 2),
                best_acc=_format_float(row["best_acc"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(out_dir, rows):
    import subprocess
    import sys

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "plot_dir": str(plot_dir),
        "rows": _json_clean(rows),
    }
    script = r"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = json.loads(sys.argv[1])
plot_dir = Path(p["plot_dir"])
rows = p["rows"]

groups = {}
for row in rows:
    groups.setdefault((row["dataset"], row["model"]), []).append(row)

for (dataset, model), items in sorted(groups.items()):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for variant in sorted({r["variant"] for r in items}):
        pts = [r for r in items if r["variant"] == variant]
        pts.sort(key=lambda r: float(r["compression"]))
        ax.plot(
            [float(r["compression"]) for r in pts],
            [float(r["test_acc_mean"]) for r in pts],
            marker="o",
            label=variant,
        )
    ax.set_xlabel("Actual node compression (x)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{dataset} {model} HCGC ablation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"ablation_{dataset}_{model}.png", dpi=180)
    plt.close(fig)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", script, json.dumps(payload)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"[plots] skipped: matplotlib failed in subprocess ({exc})")


def _save_outputs(out_dir, rows, run_rows, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            r["dataset"],
            r["model"],
            r["variant"],
            -float(r["ratio"]),
        ),
    )
    run_rows = sorted(
        run_rows,
        key=lambda r: (
            r["dataset"],
            r["model"],
            r["variant"],
            -float(r["ratio"]),
            int(r["run"]),
        ),
    )

    row_path = out_dir / "ablation_rows.csv"
    run_path = out_dir / "ablation_run_rows.csv"
    json_path = out_dir / "ablation_rows.json"
    manifest_path = out_dir / "ablation_manifest.json"
    summary_path = out_dir / "ablation_summary.csv"
    summary_json_path = out_dir / "ablation_summary.json"
    md_path = out_dir / "ablation_summary.md"

    _write_csv(row_path, rows, ROW_FIELDS)
    _write_csv(run_path, run_rows, RUN_FIELDS)
    payload = {
        "datasets": args.datasets,
        "models": args.models,
        "variants": args.variants,
        "ratios": args.ratios,
        "rows": rows,
        "run_rows": run_rows,
    }
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2),
        encoding="utf-8",
    )
    manifest = {
        "arguments": vars(args),
        "timed_run_seeds": [
            int(args.base_seed) + i for i in range(args.runs)
        ],
        "warmup_seed_policy": (
            "base_seed + 1,000,000 + warmup_index; discarded"
        ),
        "paper_protocol": {
            "candidate_budget": int(args.max_candidates),
            "ratio_search": args.ratio_search,
            "auto_search_runs": int(args.auto_search_runs),
            "auto_target_tolerance": args.auto_target_tolerance,
            "edge_weight_mode": args.edge_weight_mode,
        },
    }
    manifest_path.write_text(
        json.dumps(_json_clean(manifest), indent=2),
        encoding="utf-8",
    )

    summary_rows = _summary(rows)
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    summary_json_path.write_text(
        json.dumps(_json_clean(summary_rows), indent=2),
        encoding="utf-8",
    )
    _write_markdown_table(md_path, summary_rows)
    _write_plots(out_dir, rows)

    print(f"[save] rows    : {row_path}")
    print(f"[save] runs    : {run_path}")
    print(f"[save] data    : {json_path}")
    print(f"[save] manifest: {manifest_path}")
    print(f"[save] summary : {summary_path}")
    print(f"[save] table   : {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run HCGC ablation sweeps and save CSV/JSON/table outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["imdb", "dblp"],
                        choices=list(LOADERS))
    parser.add_argument("--models", nargs="+",
                        default=["sage", "rgcn", "gat", "appnp"],
                        choices=list(_DOWNSTREAM_MODELS))
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.5, 0.3, 0.25, 0.2, 0.15, 0.1])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        choices=sorted(VARIANTS))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Timed run i uses base_seed + i")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default="data")
    parser.add_argument("--plot-dir", default="ablation_results")
    parser.add_argument("--ratio-search", default="precise",
                        choices=["fast", "precise"])
    parser.add_argument("--auto-search-runs", type=int, default=12)
    parser.add_argument("--auto-target-tolerance", type=float, default=0.02)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--train-hidden", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--emb-method", default="gnn",
                        choices=["gnn", "fast", "relprop", "metapath2vec"])
    parser.add_argument("--raw-no-l2", action="store_true")
    parser.add_argument("--relprop-hops", type=int, default=2)
    parser.add_argument("--relprop-outdim", type=int, default=128)
    parser.add_argument("--edge-weight-mode", default="binary",
                        choices=["binary", "count", "log_count", "density"])
    parser.add_argument("--merge-objective", default="ward",
                        choices=["ward", "quotient_de"])
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip dataset/model/variant groups already present "
                             "with all requested ratios in ablation_rows.csv.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned jobs without running them.")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1")

    args.ratios = sorted(args.ratios, reverse=True)
    out_dir = Path(args.plot_dir)
    row_path = out_dir / "ablation_rows.csv"
    run_path = out_dir / "ablation_run_rows.csv"
    rows = _load_existing(row_path) if args.resume else []
    run_rows = _load_existing(run_path) if args.resume else []
    done = _completed_keys(rows, args.ratios) if args.resume else set()

    jobs = []
    for dataset in args.datasets:
        for model in args.models:
            for variant in args.variants:
                key = (dataset, model, variant)
                if key in done:
                    continue
                jobs.append(key)

    print("=" * 80)
    print("HCGC Ablation Sweep")
    print("=" * 80)
    print(f"datasets : {args.datasets}")
    print(f"models   : {args.models}")
    print(f"variants : {args.variants}")
    print(f"ratios   : {args.ratios}")
    print(f"runs     : {args.warmup} warmup + {args.runs} timed")
    print(f"seeds    : {args.base_seed} .. "
          f"{args.base_seed + args.runs - 1}")
    print(f"max cand : {args.max_candidates}")
    print(f"search   : {args.ratio_search}, runs={args.auto_search_runs}, "
          f"tolerance={args.auto_target_tolerance}")
    print(f"output   : {out_dir}")
    print(f"jobs     : {len(jobs)}")
    print("=" * 80)

    if args.dry_run:
        for idx, (dataset, model, variant) in enumerate(jobs, 1):
            cfg = _variant_config(variant, args)
            print(f"{idx:03d}: dataset={dataset} model={model} "
                  f"variant={variant} flags={cfg}")
        return

    baseline_cache = {}
    total = len(jobs)
    for idx, (dataset, model, variant) in enumerate(jobs, 1):
        cfg = _variant_config(variant, args)
        print("\n" + "#" * 80)
        print(f"Job {idx}/{total}: dataset={dataset} model={model} "
              f"variant={variant} ({VARIANTS[variant]['label']})")
        print("#" * 80)

        base_key = (dataset, model)
        base_override = baseline_cache.get(base_key)
        run_baseline = (not args.no_baseline) and base_override is None
        warmup_now = args.warmup if base_override is None else 0

        base_stats, sweep = run_sweep(
            dataset=dataset,
            ratios=args.ratios,
            runs=args.runs,
            warmup=warmup_now,
            device=args.device,
            root=args.root,
            pretrain=cfg["pretrain"],
            train_epochs=args.train_epochs,
            train_hidden=args.train_hidden,
            mini_batch_size=args.mini_batch_size,
            model_name=model,
            baseline=run_baseline,
            baseline_stats=base_override,
            emb_method=cfg["emb_method"],
            coarsen_l2_normalize=cfg["coarsen_l2_normalize"],
            relprop_hops=args.relprop_hops,
            relprop_outdim=args.relprop_outdim,
            type_thresholds=cfg["type_thresholds"],
            metapath_thresholds=cfg["metapath_thresholds"],
            edge_weight_mode=cfg["edge_weight_mode"],
            pairwise_merge=cfg["pairwise_merge"],
            merge_objective=cfg["merge_objective"],
            skip_reassignment=cfg["skip_reassignment"],
            compressor=cfg["compressor"],
            ratio_search=args.ratio_search,
            auto_search_runs=args.auto_search_runs,
            auto_target_tolerance=args.auto_target_tolerance,
            max_candidates=args.max_candidates,
            base_seed=args.base_seed,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_patience=args.pretrain_patience,
        )
        if (not args.no_baseline) and base_key not in baseline_cache:
            baseline_cache[base_key] = base_stats

        rows = [
            r for r in rows
            if not (r["dataset"] == dataset
                    and r["model"] == model
                    and r["variant"] == variant)
        ]
        rows.extend(
            _row_from_entry(dataset, model, variant, cfg, base_stats, entry, args)
            for entry in sweep
        )
        run_rows = [
            r for r in run_rows
            if not (r["dataset"] == dataset
                    and r["model"] == model
                    and r["variant"] == variant)
        ]
        for entry in sweep:
            run_rows.extend(
                _run_rows_from_entry(
                    dataset, model, variant, cfg, entry, args)
            )
        _save_outputs(out_dir, rows, run_rows, args)

    print("\nCompleted ablation sweep.")


if __name__ == "__main__":
    main()
