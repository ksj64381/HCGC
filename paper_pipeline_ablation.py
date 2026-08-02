#!/usr/bin/env python
"""Run manuscript-faithful HCGC / CGC-Homo measurements.

The runner reuses this repository's dataset loaders, HCGC phase functions,
downstream training, evaluation, and output writers. It can also target another
checkout through ``--hcgc-root``. The experimental knobs added here are:

  - max_candidates for both HCGC and CGC-Homo
  - representation used by HCGC and CGC-Homo

For embedded CGC-Homo, the HCGC pretrain context is built once and shared with
HCGC. The reported CGC-Homo-Emb compression time includes that shared embedding
time, but the embedding is not recomputed just for CGC-Homo.

Each timed run also records its seed, achieved-ratio error, stage timings,
CUDA peak memory, process RSS, storage, training diagnostics, and provenance.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return False


def git_checkout_metadata(root: Path) -> Dict[str, Any]:
    """Return the commit and dirty state used for an experiment run."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def runtime_environment_metadata() -> Dict[str, Any]:
    """Collect enough environment detail to make a long sweep auditable."""
    info: Dict[str, Any] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch_geometric": _package_version("torch-geometric"),
    }
    try:
        import torch

        info.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "cudnn": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available() else None
            ),
        })
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(idx).total_memory),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(idx)),
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return info


def dataset_metadata(data, target_type: str) -> Dict[str, Any]:
    node_counts = {
        str(nt): int(data[nt].num_nodes) for nt in data.node_types
    }
    edge_counts = {
        "|".join(map(str, et)): int(data[et].edge_index.shape[1])
        for et in data.edge_types
    }
    feature_dims = {}
    for nt in data.node_types:
        x = getattr(data[nt], "x", None)
        feature_dims[str(nt)] = int(x.shape[1]) if x is not None else None

    target = data[target_type]
    split_counts = {}
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(target, name, None)
        split_counts[name.removesuffix("_mask")] = (
            int(mask.sum().item()) if mask is not None else None
        )
    y = getattr(target, "y", None)
    valid_y = y[y >= 0] if y is not None else None
    num_classes = (
        int(valid_y.max().item()) + 1
        if valid_y is not None and valid_y.numel() else None
    )
    return {
        "total_nodes": int(sum(node_counts.values())),
        "total_directed_edge_entries": int(sum(edge_counts.values())),
        "target_type": str(target_type),
        "target_nodes": int(target.num_nodes),
        "target_num_classes": num_classes,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "feature_dims": feature_dims,
        "target_split_counts": split_counts,
    }


def process_memory_snapshot() -> Dict[str, Optional[int]]:
    """Return current RSS and the process-wide maximum RSS seen so far."""
    current_rss = None
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            rss_pages = int(f.read().split()[1])
        current_rss = rss_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, AttributeError, IndexError):
        try:
            import psutil

            current_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            pass

    max_rss = None
    try:
        import resource

        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        max_rss = raw if sys.platform == "darwin" else raw * 1024
    except (ImportError, AttributeError, ValueError):
        pass
    return {
        "process_rss_after_run_bytes": current_rss,
        "process_max_rss_so_far_bytes": max_rss,
    }


def _cuda_device(device_str: str):
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if str(device_str).lower() == "auto":
            return torch.device("cuda")
        device = torch.device(device_str)
        return device if device.type == "cuda" else None
    except Exception:
        return None


def start_stage_profile(device_str: str) -> Dict[str, Any]:
    """Start a synchronized wall-time and CUDA peak-memory measurement."""
    state: Dict[str, Any] = {"cuda_device": _cuda_device(device_str)}
    device = state["cuda_device"]
    if device is not None:
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        state["gpu_start_allocated_bytes"] = int(
            torch.cuda.memory_allocated(device))
        state["gpu_start_reserved_bytes"] = int(
            torch.cuda.memory_reserved(device))
    state["started_at"] = time.perf_counter()
    return state


def finish_stage_profile(state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    device = state.get("cuda_device")
    profile: Dict[str, Any] = {
        "gpu_start_allocated_bytes": state.get("gpu_start_allocated_bytes"),
        "gpu_start_reserved_bytes": state.get("gpu_start_reserved_bytes"),
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "gpu_peak_allocated_delta_bytes": None,
        "gpu_peak_reserved_delta_bytes": None,
    }
    if device is not None:
        import torch

        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        start_allocated = int(state["gpu_start_allocated_bytes"])
        start_reserved = int(state["gpu_start_reserved_bytes"])
        profile.update({
            "gpu_peak_allocated_bytes": peak_allocated,
            "gpu_peak_reserved_bytes": peak_reserved,
            "gpu_peak_allocated_delta_bytes": max(
                peak_allocated - start_allocated, 0),
            "gpu_peak_reserved_delta_bytes": max(
                peak_reserved - start_reserved, 0),
        })
    elapsed = time.perf_counter() - state["started_at"]
    return elapsed, profile


def prefixed_profile(prefix: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {f"gpu_{prefix}_{key.removeprefix('gpu_')}": value
            for key, value in profile.items()}


def max_available(*values: Any) -> Optional[int]:
    available = [int(value) for value in values if value is not None]
    return max(available) if available else None


def resolve_hcgc_root(raw: Optional[str]) -> Path:
    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    env_root = os.environ.get("HCGC_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([
        SCRIPT_DIR,
        Path.cwd(),
        SCRIPT_DIR.parent / "HCGC",
        SCRIPT_DIR.parent / "HCGC_original",
        SCRIPT_DIR.parent.parent / "HCGC",
        SCRIPT_DIR.parent.parent / "HCGC_original",
    ])

    seen = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "experiments.py").exists() and (root / "benchmark.py").exists():
            return root

    checked = "\n  ".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        "Could not find an HCGC repository root. Pass --hcgc-root or set "
        f"HCGC_ROOT. Checked:\n  {checked}"
    )


def import_original_pipeline(hcgc_root: Path) -> Dict[str, Any]:
    """Import HCGC modules from the selected checkout without stale imports."""
    filtered: List[str] = []
    for item in sys.path:
        path = Path(item or ".")
        if _same_path(path, SCRIPT_DIR):
            continue
        filtered.append(item)

    root_str = str(hcgc_root)
    sys.path[:] = [root_str] + [
        p for p in filtered if not _same_path(Path(p or "."), hcgc_root)
    ]

    for name in list(sys.modules):
        if (
            name == "benchmark"
            or name == "experiments"
            or name == "hcgc_module"
            or name == "hcgc"
            or name.startswith("hcgc.")
        ):
            sys.modules.pop(name, None)

    modules = {
        "experiments": importlib.import_module("experiments"),
        "benchmark": importlib.import_module("benchmark"),
        "api": importlib.import_module("hcgc._api"),
        "pipeline": importlib.import_module("hcgc._pipeline"),
        "coarsen": importlib.import_module("hcgc._coarsen"),
        "baselines": importlib.import_module("hcgc._baselines"),
        "config": importlib.import_module("hcgc._config"),
    }

    exp_path = Path(getattr(modules["experiments"], "__file__", "")).resolve()
    if not _same_path(exp_path.parent, hcgc_root):
        raise RuntimeError(
            f"Imported experiments.py from {exp_path}, expected under {hcgc_root}"
        )
    return modules


RAW_EMB_METHOD = "raw"
EMB_METHOD_CHOICES = ("raw", "gnn", "fast", "relprop", "metapath2vec")
PAPER_HCGC_MAX_CANDIDATES = 128
PAPER_CGC_HOMO_MAX_CANDIDATES = 128


def method_labels(compressors: Sequence[str],
                  emb_methods: Sequence[str]) -> List[str]:
    emb_methods = [str(m).lower() for m in emb_methods]
    learned_methods = [m for m in emb_methods if m != RAW_EMB_METHOD]
    labels: List[str] = []
    for compressor in compressors:
        comp = str(compressor).lower()
        if comp == "hcgc":
            if RAW_EMB_METHOD in emb_methods:
                labels.append("hcgc_raw")
            for method in learned_methods:
                labels.append(f"hcgc_emb:{method}")
        elif comp == "cgc_homo":
            labels.append("cgc_homo_raw")
            for method in learned_methods:
                labels.append(f"cgc_homo_emb:{method}")
        else:
            raise ValueError(
                "paper_pipeline_ablation.py supports compressors: "
                "hcgc and cgc_homo"
            )
    return labels


def method_kind(label: str) -> str:
    return label.split(":", 1)[0]


def method_emb(label: str) -> Optional[str]:
    return label.split(":", 1)[1] if ":" in label else None


def display_method(label: str, all_methods: Optional[Sequence[str]] = None) -> str:
    kind = method_kind(label)
    emb = method_emb(label)
    learned = []
    if all_methods:
        learned = sorted({
            m.split(":", 1)[1]
            for m in all_methods
            if ":" in m and method_kind(m) in ("hcgc_emb", "cgc_homo_emb")
        })
    suffix = ""
    if emb and len(learned) > 1:
        suffix = f"-{emb.upper()}"
    names = {
        "hcgc_emb": f"HCGC-Emb{suffix}",
        "hcgc_raw": "HCGC-Raw",
        "cgc_homo_raw": "CGC-Homo",
        "cgc_homo_emb": f"CGC-Homo-Emb{suffix}",
    }
    return names.get(kind, label.replace(":", "-"))


def resolve_dataset_name(dataset: str, acm_variant: str) -> str:
    name = str(dataset).lower()
    if name in {"paper_acm", "acm_paper"}:
        return "acm3"
    if name in {"full_acm", "acm_full"}:
        return "acm"
    if name == "acm":
        return "acm3" if str(acm_variant).lower() == "paper" else "acm"
    return name


def resolve_datasets(datasets: Sequence[str],
                     acm_variant: str) -> Tuple[List[str], List[Dict[str, str]]]:
    resolved: List[str] = []
    aliases: List[Dict[str, str]] = []
    for raw in datasets:
        mapped = resolve_dataset_name(raw, acm_variant)
        resolved.append(mapped)
        if str(raw).lower() != mapped:
            aliases.append({"requested": str(raw), "resolved": mapped})
    return resolved, aliases


def validate_datasets(datasets: Sequence[str], loaders: Dict[str, Any]) -> None:
    missing = [d for d in datasets if d not in loaders]
    if missing:
        raise ValueError(
            f"Unknown dataset(s): {missing}. Available: {sorted(loaders)}"
        )


def initialize_hcgc_state(mods: Dict[str, Any], data, target_type: str,
                          device: str):
    import torch

    cfg_mod = mods["config"]
    api = mods["api"]
    if device == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = str(device)
    cfg_mod.set_device(dev_str)

    data = api._ensure_node_features(data)
    cfg_mod._CFG.node_types = list(data.node_types)
    cfg_mod._CFG.target_type = api._detect_target_type(data, target_type)
    cfg_mod._CFG.num_classes = (
        int(data[cfg_mod._CFG.target_type].y.max().item()) + 1
    )
    cfg_mod._CFG.dataset = None
    return data, cfg_mod._CFG.target_type, dev_str


def effective_target_tolerance(args) -> float:
    if args.auto_target_tolerance is not None:
        return float(args.auto_target_tolerance)
    return 0.05 if str(args.ratio_search).lower() == "precise" else 0.15


def make_hcgc_args(mods: Dict[str, Any], ratio: float, pretrain: bool,
                   emb_method: str, args, max_candidates: int,
                   run_seed: int):
    built = mods["api"]._build_args(
        ratio=ratio,
        pretrain=pretrain,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_patience=args.pretrain_patience,
        emb_method=emb_method if pretrain else "gnn",
        coarsen_l2_normalize=not args.raw_no_l2,
        relprop_hops=args.relprop_hops,
        relprop_outdim=args.relprop_outdim,
        mini_batch_size=args.mini_batch_size,
        use_soft_labels=False,
        pairwise_merge=args.pairwise_merge,
        merge_objective=args.merge_objective,
        skip_reassignment=args.skip_reassignment,
        type_thresholds=args.type_thresholds,
        metapath_thresholds=args.metapath_thresholds,
        ratio_search=args.ratio_search,
        auto_search_runs=args.auto_search_runs,
        auto_target_tolerance=effective_target_tolerance(args),
    )
    built.max_candidates = int(max_candidates)
    built.base_seed = int(run_seed)
    return built


def load_hcgc_context(mods: Dict[str, Any], data, target_type: str,
                      ratio: float, pretrain: bool, emb_method: str, args,
                      max_candidates: int, run_seed: int):
    data, target_type, _ = initialize_hcgc_state(
        mods, data, target_type, args.device)
    hcgc_args = make_hcgc_args(
        mods, ratio, pretrain=pretrain, emb_method=emb_method, args=args,
        max_candidates=max_candidates, run_seed=run_seed)
    stage = start_stage_profile(args.device)
    ctx = mods["pipeline"]._load_and_pretrain(data, hcgc_args)
    ctx_time, profile = finish_stage_profile(stage)
    return ctx, hcgc_args, ctx_time, profile


def result_info_from_stats(stats: Dict[str, Any], n_orig: int, n_comp: int,
                           t_coarsen: float, t_build: float,
                           emb_diag: Dict[str, float],
                           extra: Dict[str, Any]) -> Dict[str, Any]:
    actual_ratio = n_comp / max(n_orig, 1)
    info = {
        "compression": round(1.0 / max(actual_ratio, 1e-12), 4),
        "n_nodes_orig": n_orig,
        "n_nodes_comp": n_comp,
        "coarsen_time": float(t_coarsen),
        "build_time": float(t_build),
        "nodes_orig": stats["nodes_orig"],
        "nodes_comp": stats["nodes_comp"],
        "edges_orig": stats["edges_orig"],
        "edges_comp": stats["edges_comp"],
        "edge_ratio": round(stats["edge_ratio"], 4),
        "freeze_node_types": [],
        "target_emb_distortion": emb_diag["distortion"],
        "target_emb_cosine": emb_diag["cosine"],
    }
    info.update(extra)
    return info


def compress_hcgc_from_context(mods: Dict[str, Any], data, target_type: str,
                               ctx, hcgc_args, args, method_label: str):
    api = mods["api"]
    coarsen = mods["coarsen"]
    cfg = mods["config"]._CFG

    stage = start_stage_profile(args.device)
    cm, t_coarsen = mods["pipeline"]._coarsen_from_context(ctx, hcgc_args)
    cm = api._apply_freeze_node_types(
        cm, ctx["offsets"], ctx["type_boundaries"], cfg.node_types,
        None, False)

    t_build0 = time.perf_counter()
    cdata, local_cm, stats = coarsen.build_compressed_data(
        data,
        cm,
        ctx["offsets"],
        ctx["type_boundaries"],
        use_soft_labels=False,
        emb_dict=ctx["emb_dict"] if hcgc_args.use_emb_coarsen else None,
        edge_weight_mode=args.edge_weight_mode,
    )
    t_build = time.perf_counter() - t_build0
    method_wall, profile = finish_stage_profile(stage)

    emb_diag = api._target_embedding_diagnostics(
        ctx.get("emb_dict"), local_cm, cfg.target_type)
    n_orig = int(ctx["type_boundaries"][-1])
    n_comp = int(stats["nodes_comp"])
    info = result_info_from_stats(
        stats,
        n_orig,
        n_comp,
        t_coarsen,
        t_build,
        emb_diag,
        {
            "compressor": method_label,
            "ratio_search": args.ratio_search,
            "auto_search_runs": int(args.auto_search_runs),
            "auto_target_tolerance": effective_target_tolerance(args),
            "max_candidates": int(hcgc_args.max_candidates),
            "coarsening_edge_input": dict(ctx.get("edge_input_stats", {})),
            "embedding_method": (
                str(hcgc_args.emb_method)
                if hcgc_args.use_emb_coarsen else RAW_EMB_METHOD
            ),
            "ratio_search_diagnostics": dict(
                ctx.get("last_auto_coarsen_info") or {}),
        },
    )
    return api.HCGCResult(
        data=cdata,
        ratio=n_comp / max(n_orig, 1),
        node_map=local_cm,
        info=info,
    ), method_wall, profile


def embedding_representation(ctx, data, mods: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    flat = ctx.get("coarsen_features")
    feat_dims = ctx.get("coarsen_feat_dims")
    type_boundaries = ctx.get("type_boundaries")
    rep_dict: Dict[str, Any] = {}
    pos = 0
    prev = 0
    if flat is not None and feat_dims is not None and type_boundaries is not None:
        for i, nt in enumerate(mods["config"]._CFG.node_types):
            end = int(type_boundaries[i])
            n = end - prev
            dim = int(feat_dims[i])
            size = n * dim
            arr = np.asarray(flat[pos:pos + size], dtype=np.float32).reshape(n, dim)
            rep_dict[nt] = torch.from_numpy(arr.copy()).float().contiguous()
            pos += size
            prev = end
        return rep_dict

    emb_dict = ctx.get("emb_dict") or {}
    for nt in mods["config"]._CFG.node_types:
        if nt in emb_dict:
            rep_dict[nt] = emb_dict[nt].detach().float().cpu().contiguous()
            continue
        rep_dict[nt] = mods["baselines"]._node_representation(data, nt)
    return rep_dict


def compress_cgc_homo_emb_from_context(mods: Dict[str, Any], data,
                                       target_type: str, ctx, ratio: float,
                                       args, max_candidates: int,
                                       emb_method: str):
    baselines = mods["baselines"]
    coarsen = mods["coarsen"]
    api = mods["api"]
    cfg = mods["config"]._CFG

    stage = start_stage_profile(args.device)
    rep_dict = embedding_representation(ctx, data, mods)
    t_coarse0 = time.perf_counter()
    adj, degree, type_id = baselines._homogeneous_adjacency(
        data, ctx["offsets"], ctx["type_boundaries"])
    cm = baselines._cgc_homo_partition(
        data,
        ctx["offsets"],
        ctx["type_boundaries"],
        rep_dict,
        adj,
        degree,
        type_id,
        ratio=ratio,
        max_hub_degree=args.max_hub_degree,
        max_candidates=max_candidates,
    )
    t_coarsen = time.perf_counter() - t_coarse0

    t_build0 = time.perf_counter()
    cdata, local_cm, stats = coarsen.build_compressed_data(
        data,
        cm,
        ctx["offsets"],
        ctx["type_boundaries"],
        use_soft_labels=False,
        emb_dict=rep_dict,
        edge_weight_mode=args.edge_weight_mode,
    )
    t_build = time.perf_counter() - t_build0
    method_wall, profile = finish_stage_profile(stage)

    emb_diag = api._target_embedding_diagnostics(
        rep_dict, local_cm, cfg.target_type)
    n_orig = int(ctx["type_boundaries"][-1])
    n_comp = int(stats["nodes_comp"])
    info = result_info_from_stats(
        stats,
        n_orig,
        n_comp,
        t_coarsen,
        t_build,
        emb_diag,
        {
            "compressor": "cgc_homo_emb",
            "max_hub_degree": int(args.max_hub_degree),
            "max_candidates": int(max_candidates),
            "embedding_method": emb_method,
        },
    )
    return api.HCGCResult(
        data=cdata,
        ratio=n_comp / max(n_orig, 1),
        node_map=local_cm,
        info=info,
    ), method_wall, profile


def compress_cgc_homo_raw(mods: Dict[str, Any], data, target_type: str,
                          ratio: float, args, max_candidates: int):
    stage = start_stage_profile(args.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mods["benchmark"].compress_cgc_homo(
            data,
            ratio=ratio,
            target_type=target_type,
            edge_weight_mode=args.edge_weight_mode,
            use_soft_labels=False,
            freeze_node_types=None,
            max_hub_degree=args.max_hub_degree,
            max_candidates=max_candidates,
            verbose=False,
        )
    method_wall, profile = finish_stage_profile(stage)
    return result, method_wall, profile


def storage_and_eval_record(mods: Dict[str, Any], result, data,
                            target_type: str, model_name: str,
                            method_label: str, ratio: float,
                            t_compress: float,
                            embedding_time: float,
                            embedding_profile: Dict[str, Any],
                            compression_profile: Dict[str, Any],
                            run_seed: int,
                            args,
                            all_method_keys: Sequence[str]) -> Dict[str, Any]:
    benchmark = mods["benchmark"]
    oracle = benchmark.oracle_upper_bound(result, data, target_type, "test_mask")
    oracle_val = benchmark.oracle_upper_bound(result, data, target_type, "val_mask")

    mods["config"].set_seed(int(run_seed))
    train_stage = start_stage_profile(args.device)
    test_metrics, t_train_internal = benchmark.train_downstream(
        result,
        data,
        target_type,
        device_str=args.device,
        epochs=args.train_epochs,
        hidden=args.train_hidden,
        mini_batch_size=args.mini_batch_size,
        model_name=model_name,
        eval_protocol="original",
        use_soft_labels=False,
        use_edge_weights=str(args.edge_weight_mode).lower() != "binary",
        return_metrics=True,
    )
    t_train_wall, training_profile = finish_stage_profile(train_stage)

    n_orig = result.info["n_nodes_orig"]
    n_comp = result.info["n_nodes_comp"]
    e_orig = result.info["edges_orig"]
    e_comp = result.info["edges_comp"]
    storage_orig = benchmark._heterodata_storage_bytes(data)
    storage_comp_graph = benchmark._heterodata_storage_bytes(result.data)
    storage_map = benchmark._node_map_storage_bytes(result.node_map)
    storage_comp = storage_comp_graph + storage_map
    test_acc = test_metrics["accuracy"]
    edge_input = result.info.get("coarsening_edge_input", {})
    search_info = result.info.get("ratio_search_diagnostics", {}) or {}
    achieved_ratio = n_comp / max(n_orig, 1)
    achieved_compression = 1.0 / max(achieved_ratio, 1e-12)
    target_compression = 1.0 / max(float(ratio), 1e-12)
    compressed_node_counts = {
        str(nt): int(result.data[nt].num_nodes)
        for nt in result.data.node_types
    }
    compressed_edge_counts = {
        "|".join(map(str, et)): int(result.data[et].edge_index.shape[1])
        for et in result.data.edge_types
    }
    compressed_split_counts = {}
    if target_type in result.data.node_types:
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(result.data[target_type], name, None)
            compressed_split_counts[name.removesuffix("_mask")] = (
                int(mask.sum().item()) if mask is not None else None
            )

    record = {
        "seed": int(run_seed),
        "target_retention_ratio": float(ratio),
        "achieved_retention_ratio": achieved_ratio,
        "retention_ratio_abs_error": abs(achieved_ratio - float(ratio)),
        "retention_ratio_relative_error": (
            abs(achieved_ratio - float(ratio)) / max(float(ratio), 1e-12)
        ),
        "target_compression_factor": target_compression,
        "compression_factor_abs_error": abs(
            achieved_compression - target_compression),
        "compression_factor_relative_error": abs(
            achieved_compression - target_compression
        ) / max(target_compression, 1e-12),
        "node_ratio": achieved_ratio,
        "edge_ratio": e_comp / max(e_orig, 1),
        "compression": achieved_compression,
        "node_compression": achieved_compression,
        "edge_compression": e_orig / max(e_comp, 1),
        "storage_orig_bytes": storage_orig,
        "storage_comp_graph_bytes": storage_comp_graph,
        "storage_map_bytes": storage_map,
        "storage_comp_bytes": storage_comp,
        "storage_ratio": storage_comp / max(storage_orig, 1),
        "storage_compression": storage_orig / max(storage_comp, 1),
        "storage_reduction": 1.0 - (storage_comp / max(storage_orig, 1)),
        "t_compress": t_compress,
        "t_coarsen": result.info["coarsen_time"],
        "t_build": result.info.get("build_time"),
        "t_coarsen_build_wall": max(t_compress - embedding_time, 0.0),
        "t_train": t_train_internal,
        "t_train_internal": t_train_internal,
        "t_train_wall": t_train_wall,
        "t_total": t_compress + t_train_internal,
        "t_total_wall": t_compress + t_train_wall,
        "embedding_time": embedding_time,
        "t_embedding_context": embedding_time,
        "test_acc": test_acc,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_support": test_metrics["support"],
        "resplit_triggered": test_metrics.get("resplit_triggered", False),
        "val_supernodes_before_resplit": test_metrics.get(
            "val_supernodes_before_resplit"),
        "val_supernodes_after_resplit": test_metrics.get(
            "val_supernodes_after_resplit"),
        "training_mode": test_metrics.get("training_mode"),
        "epochs_requested": test_metrics.get(
            "epochs_requested", int(args.train_epochs)),
        "epochs_ran": test_metrics.get("epochs_ran"),
        "early_stopped": test_metrics.get("early_stopped"),
        "best_val_acc": test_metrics.get("best_val_acc"),
        "oracle_acc": oracle["oracle_acc"],
        "oracle_val_acc": oracle_val["oracle_acc"],
        "oracle_gap": oracle["oracle_acc"] - test_acc,
        "oracle_n_nodes": oracle["oracle_n_nodes"],
        "oracle_n_supernodes": oracle["oracle_n_supernodes"],
        "oracle_mixed_frac": oracle["oracle_mixed_frac"],
        "oracle_mean_purity": oracle["oracle_mean_purity"],
        "target_emb_distortion": result.info.get(
            "target_emb_distortion", float("nan")),
        "target_emb_cosine": result.info.get(
            "target_emb_cosine", float("nan")),
        "compressor": display_method(method_label, all_method_keys),
        "method_key": method_label,
        "embedding_method": method_emb(method_label) or RAW_EMB_METHOD,
        "ratio_search": args.ratio_search,
        "ratio_search_scale_used": search_info.get("scale_used"),
        "ratio_search_threshold_mode": search_info.get("threshold_mode"),
        "ratio_search_n_coarsen_runs": search_info.get("n_coarsen_runs"),
        "ratio_search_saturated": search_info.get("saturated"),
        "ratio_search_probe_baseline": search_info.get("probe_baseline"),
        "ratio_search_probe_after": search_info.get("probe_after"),
        "ratio_search_probe_loss": search_info.get("probe_loss"),
        "ratio_search_all_runs_json": json.dumps(
            _json_clean(search_info.get("all_runs", [])),
            separators=(",", ":")),
        "ratio": ratio,
        "n_nodes_orig": n_orig,
        "n_nodes_comp": n_comp,
        "edges_orig": e_orig,
        "edges_comp": e_comp,
        "compressed_node_counts_json": json.dumps(
            compressed_node_counts, sort_keys=True),
        "compressed_edge_counts_json": json.dumps(
            compressed_edge_counts, sort_keys=True),
        "compressed_target_split_counts_json": json.dumps(
            compressed_split_counts, sort_keys=True),
        "coarsening_input_edge_entries": edge_input.get(
            "input_edge_entries"),
        "coarsening_duplicate_entries_removed": edge_input.get(
            "duplicate_entries_removed"),
        "coarsening_self_loops_removed": edge_input.get(
            "self_loops_removed"),
        "coarsening_unique_undirected_edges": edge_input.get(
            "unique_undirected_edges"),
    }
    record.update(prefixed_profile("embedding", embedding_profile))
    record.update(prefixed_profile("coarsen_build", compression_profile))
    record.update(prefixed_profile("training", training_profile))
    record.update({
        "gpu_peak_allocated_total_bytes": max_available(
            embedding_profile.get("gpu_peak_allocated_bytes"),
            compression_profile.get("gpu_peak_allocated_bytes"),
            training_profile.get("gpu_peak_allocated_bytes")),
        "gpu_peak_reserved_total_bytes": max_available(
            embedding_profile.get("gpu_peak_reserved_bytes"),
            compression_profile.get("gpu_peak_reserved_bytes"),
            training_profile.get("gpu_peak_reserved_bytes")),
        "gpu_max_stage_allocated_delta_bytes": max_available(
            embedding_profile.get("gpu_peak_allocated_delta_bytes"),
            compression_profile.get("gpu_peak_allocated_delta_bytes"),
            training_profile.get("gpu_peak_allocated_delta_bytes")),
        "gpu_max_stage_reserved_delta_bytes": max_available(
            embedding_profile.get("gpu_peak_reserved_delta_bytes"),
            compression_profile.get("gpu_peak_reserved_delta_bytes"),
            training_profile.get("gpu_peak_reserved_delta_bytes")),
    })
    record.update(process_memory_snapshot())
    return record


def run_methods_once(mods: Dict[str, Any], data, target_type: str,
                     model_name: str, ratio: float, method_keys: Sequence[str],
                     args, max_candidates: int,
                     run_seed: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    mods["config"].set_seed(int(run_seed))

    learned_needed = sorted({
        method_emb(k)
        for k in method_keys
        if method_kind(k) in ("hcgc_emb", "cgc_homo_emb")
    })
    emb_contexts: Dict[
        str, Tuple[Any, Any, float, Dict[str, Any]]
    ] = {}
    for emb_method in learned_needed:
        emb_contexts[emb_method] = load_hcgc_context(
            mods, data, target_type, ratio, True, emb_method,
            args, max_candidates, run_seed)

    raw_ctx = raw_args = None
    raw_ctx_time = 0.0
    raw_ctx_profile: Dict[str, Any] = {}
    if "hcgc_raw" in method_keys:
        raw_ctx, raw_args, raw_ctx_time, raw_ctx_profile = load_hcgc_context(
            mods, data, target_type, ratio, False, RAW_EMB_METHOD,
            args, max_candidates, run_seed)

    for method_key in method_keys:
        kind = method_kind(method_key)
        emb_method = method_emb(method_key)
        if kind == "hcgc_emb":
            emb_ctx, emb_args, emb_time, emb_profile = emb_contexts[emb_method]
            result, method_wall, compression_profile = compress_hcgc_from_context(
                mods, data, target_type, emb_ctx, emb_args, args, method_key)
            t_compress = emb_time + method_wall
            t_emb = emb_time
        elif kind == "hcgc_raw":
            emb_profile = raw_ctx_profile
            result, method_wall, compression_profile = compress_hcgc_from_context(
                mods, data, target_type, raw_ctx, raw_args, args, method_key)
            t_compress = raw_ctx_time + method_wall
            t_emb = 0.0
        elif kind == "cgc_homo_raw":
            emb_profile = {}
            result, t_compress, compression_profile = compress_cgc_homo_raw(
                mods, data, target_type, ratio, args, max_candidates)
            t_emb = 0.0
        elif kind == "cgc_homo_emb":
            emb_ctx, _, emb_time, emb_profile = emb_contexts[emb_method]
            result, method_wall, compression_profile = compress_cgc_homo_emb_from_context(
                mods, data, target_type, emb_ctx, ratio, args,
                max_candidates, emb_method)
            t_compress = emb_time + method_wall
            t_emb = emb_time
        else:
            raise ValueError(f"Unsupported method: {method_key}")

        record = storage_and_eval_record(
            mods,
            result,
            data,
            target_type,
            model_name,
            method_key,
            ratio,
            t_compress,
            t_emb,
            emb_profile,
            compression_profile,
            run_seed,
            args,
            method_keys,
        )
        records.append(record)
    return records


def mean_std(records: Sequence[Dict[str, Any]], key: str) -> Tuple[float, float]:
    values = []
    for record in records:
        try:
            values.append(float(record.get(key, float("nan"))))
        except (TypeError, ValueError):
            values.append(float("nan"))
    vals = np.array(values, dtype=float)
    if vals.size == 0 or np.isnan(vals).all():
        return float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def baseline_cache_config(args) -> Dict[str, Any]:
    return {
        "runs": int(args.runs),
        "base_seed": int(args.base_seed),
        "train_epochs": int(args.train_epochs),
        "train_hidden": int(args.train_hidden),
        "mini_batch_size": int(args.mini_batch_size),
    }


def load_baseline_cache(path: Optional[str], args) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        print(f"[baseline-cache] no existing cache at {cache_path}")
        return {}

    expected_cfg = baseline_cache_config(args)
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[baseline-cache] failed to read {cache_path}: {exc}")
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    skipped = 0
    for item in payload.get("items", []):
        cfg = item.get("config", {})
        dataset = item.get("dataset")
        model = item.get("model")
        stats = item.get("stats")
        if cfg != expected_cfg or not dataset or not model or not isinstance(stats, dict):
            skipped += 1
            continue
        out[(str(dataset), str(model))] = stats

    print(
        f"[baseline-cache] loaded {len(out)} matching baseline(s) "
        f"from {cache_path}"
    )
    if skipped:
        print(f"[baseline-cache] skipped {skipped} entry/entries with other settings")
    return out


def save_baseline_cache(path: Optional[str], cache: Dict[Tuple[str, str], Dict[str, Any]],
                        args) -> None:
    if not path:
        return
    cache_path = Path(path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = baseline_cache_config(args)

    preserved: List[Dict[str, Any]] = []
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for item in payload.get("items", []):
                if item.get("config") != cfg:
                    preserved.append(item)
        except Exception as exc:
            print(f"[baseline-cache] replacing unreadable cache {cache_path}: {exc}")

    items = preserved
    for (dataset, model), stats in sorted(cache.items()):
        items.append({
            "dataset": dataset,
            "model": model,
            "config": cfg,
            "stats": stats,
        })

    payload = {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": _json_clean(items),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[baseline-cache] saved {len(cache)} matching baseline(s) to {cache_path}")


def aggregate_ratio(records: Sequence[Dict[str, Any]], ratio: float,
                    base_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    acc_m, acc_s = mean_std(records, "test_acc")
    mf1_m, mf1_s = mean_std(records, "test_macro_f1")
    mif1_m, mif1_s = mean_std(records, "test_micro_f1")
    vo_m, vo_s = mean_std(records, "oracle_val_acc")
    to_m, to_s = mean_std(records, "oracle_acc")
    og_m, og_s = mean_std(records, "oracle_gap")
    ed_m, ed_s = mean_std(records, "target_emb_distortion")
    ec_m, ec_s = mean_std(records, "target_emb_cosine")
    node_ratio_m, node_ratio_s = mean_std(records, "node_ratio")
    edge_ratio_m, edge_ratio_s = mean_std(records, "edge_ratio")
    comp_m, comp_s = mean_std(records, "compression")
    edge_comp_m, edge_comp_s = mean_std(records, "edge_compression")
    storage_comp_m, storage_comp_s = mean_std(records, "storage_compression")
    storage_ratio_m, storage_ratio_s = mean_std(records, "storage_ratio")
    storage_red_m, storage_red_s = mean_std(records, "storage_reduction")
    storage_orig_m, storage_orig_s = mean_std(records, "storage_orig_bytes")
    storage_cg_m, storage_cg_s = mean_std(records, "storage_comp_graph_bytes")
    storage_map_m, storage_map_s = mean_std(records, "storage_map_bytes")
    storage_bytes_m, storage_bytes_s = mean_std(records, "storage_comp_bytes")
    tt_m, tt_s = mean_std(records, "t_total")
    tc_m, tc_s = mean_std(records, "t_compress")
    tr_m, tr_s = mean_std(records, "t_train")
    trw_m, trw_s = mean_std(records, "t_train_wall")
    tb_m, tb_s = mean_std(records, "t_build")
    tcb_m, tcb_s = mean_std(records, "t_coarsen_build_wall")
    ttw_m, ttw_s = mean_std(records, "t_total_wall")
    et_m, et_s = mean_std(records, "embedding_time")
    ratio_err_m, ratio_err_s = mean_std(
        records, "retention_ratio_abs_error")
    ratio_rel_err_m, ratio_rel_err_s = mean_std(
        records, "retention_ratio_relative_error")
    search_runs_m, search_runs_s = mean_std(
        records, "ratio_search_n_coarsen_runs")
    epochs_m, epochs_s = mean_std(records, "epochs_ran")
    gpu_alloc_m, gpu_alloc_s = mean_std(
        records, "gpu_peak_allocated_total_bytes")
    gpu_res_m, gpu_res_s = mean_std(
        records, "gpu_peak_reserved_total_bytes")
    gpu_delta_m, gpu_delta_s = mean_std(
        records, "gpu_max_stage_allocated_delta_bytes")

    entry = {
        "ratio": ratio,
        "acc_mean": acc_m, "acc_std": acc_s,
        "macro_f1_mean": mf1_m, "macro_f1_std": mf1_s,
        "micro_f1_mean": mif1_m, "micro_f1_std": mif1_s,
        "val_oracle_mean": vo_m, "val_oracle_std": vo_s,
        "oracle_mean": to_m, "oracle_std": to_s,
        "oracle_gap_mean": og_m, "oracle_gap_std": og_s,
        "emb_dist_mean": ed_m, "emb_dist_std": ed_s,
        "emb_cos_mean": ec_m, "emb_cos_std": ec_s,
        "node_ratio_mean": node_ratio_m, "node_ratio_std": node_ratio_s,
        "edge_ratio_mean": edge_ratio_m, "edge_ratio_std": edge_ratio_s,
        "comp_mean": comp_m, "comp_std": comp_s,
        "edge_comp_mean": edge_comp_m, "edge_comp_std": edge_comp_s,
        "storage_comp_mean": storage_comp_m,
        "storage_comp_std": storage_comp_s,
        "storage_ratio_mean": storage_ratio_m,
        "storage_ratio_std": storage_ratio_s,
        "storage_reduction_mean": storage_red_m,
        "storage_reduction_std": storage_red_s,
        "storage_orig_bytes_mean": storage_orig_m,
        "storage_orig_bytes_std": storage_orig_s,
        "storage_comp_graph_bytes_mean": storage_cg_m,
        "storage_comp_graph_bytes_std": storage_cg_s,
        "storage_map_bytes_mean": storage_map_m,
        "storage_map_bytes_std": storage_map_s,
        "storage_comp_bytes_mean": storage_bytes_m,
        "storage_comp_bytes_std": storage_bytes_s,
        "tt_mean": tt_m, "tt_std": tt_s,
        "tc_mean": tc_m, "tc_std": tc_s,
        "tr_mean": tr_m, "tr_std": tr_s,
        "train_wall_mean": trw_m, "train_wall_std": trw_s,
        "build_time_mean": tb_m, "build_time_std": tb_s,
        "coarsen_build_wall_mean": tcb_m,
        "coarsen_build_wall_std": tcb_s,
        "total_wall_mean": ttw_m, "total_wall_std": ttw_s,
        "embedding_time_mean": et_m,
        "embedding_time_std": et_s,
        "retention_abs_error_mean": ratio_err_m,
        "retention_abs_error_std": ratio_err_s,
        "retention_relative_error_mean": ratio_rel_err_m,
        "retention_relative_error_std": ratio_rel_err_s,
        "ratio_search_runs_mean": search_runs_m,
        "ratio_search_runs_std": search_runs_s,
        "epochs_ran_mean": epochs_m,
        "epochs_ran_std": epochs_s,
        "gpu_peak_allocated_bytes_mean": gpu_alloc_m,
        "gpu_peak_allocated_bytes_std": gpu_alloc_s,
        "gpu_peak_reserved_bytes_mean": gpu_res_m,
        "gpu_peak_reserved_bytes_std": gpu_res_s,
        "gpu_max_stage_allocated_delta_bytes_mean": gpu_delta_m,
        "gpu_max_stage_allocated_delta_bytes_std": gpu_delta_s,
    }
    if base_stats is not None:
        entry["acc_drop"] = acc_m - base_stats["acc_mean"]
        entry["macro_f1_drop"] = (
            mf1_m - base_stats.get("macro_f1_mean", float("nan"))
        )
        entry["train_speedup"] = base_stats["t_mean"] / max(tr_m, 1e-6)
    return entry


def run_dataset_sweeps(mods: Dict[str, Any], dataset: str,
                       method_keys: Sequence[str], ratios: Sequence[float],
                       args, max_candidates: int,
                       variant_dir: Path,
                       baseline_cache: Optional[
                           Dict[Tuple[str, str], Dict[str, Any]]
                       ] = None) -> List[Dict[str, Any]]:
    experiments = mods["experiments"]
    benchmark = mods["benchmark"]

    print(f"Loading {dataset} with original loader ...")
    data, target_type = experiments.LOADERS[dataset](args.root)
    data = experiments._add_degree_features(data)
    dataset_info = dataset_metadata(data, target_type)

    n_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    n_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    print(f"  node types : {list(data.node_types)}")
    print(f"  total nodes: {n_nodes:,}   total edges: {n_edges:,}")
    print(f"  target type: {target_type!r}")

    rows: List[Dict[str, Any]] = []
    all_results: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    for model_name in args.models:
        base_stats = None
        cache_key = (dataset, model_name)
        if args.no_baseline:
            base_stats = None
        elif baseline_cache is not None and cache_key in baseline_cache:
            base_stats = baseline_cache[cache_key]
            print(
                f"\nBaseline: reusing original graph result "
                f"model={model_name} acc={base_stats['acc_mean']:.4f} "
                f"macro_f1={base_stats.get('macro_f1_mean', float('nan')):.4f} "
                f"t={base_stats['t_mean']:.1f}s"
            )
        else:
            print(f"\nBaseline: model={model_name} original graph ...")
            base_recs = []
            for run_idx in range(args.runs):
                run_seed = int(args.base_seed) + run_idx
                print(f"  baseline run {run_idx + 1}/{args.runs} ... ",
                      end="", flush=True)
                mods["config"].set_seed(run_seed)
                metrics, elapsed = benchmark.run_baseline(
                    data,
                    target_type,
                    args.device,
                    train_epochs=args.train_epochs,
                    train_hidden=args.train_hidden,
                    mini_batch_size=args.mini_batch_size,
                    model_name=model_name,
                    return_metrics=True,
                )
                base_recs.append({
                    "test_acc": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_micro_f1": metrics["micro_f1"],
                    "t_train": elapsed,
                    "seed": run_seed,
                    "epochs_ran": metrics.get("epochs_ran"),
                    "best_val_acc": metrics.get("best_val_acc"),
                })
                print(f"t={elapsed:.1f}s acc={metrics['accuracy']:.4f}")
            base_stats = {
                "acc_mean": float(np.mean([r["test_acc"] for r in base_recs])),
                "acc_std": float(np.std([r["test_acc"] for r in base_recs])),
                "macro_f1_mean": float(np.mean(
                    [r["test_macro_f1"] for r in base_recs])),
                "macro_f1_std": float(np.std(
                    [r["test_macro_f1"] for r in base_recs])),
                "micro_f1_mean": float(np.mean(
                    [r["test_micro_f1"] for r in base_recs])),
                "micro_f1_std": float(np.std(
                    [r["test_micro_f1"] for r in base_recs])),
                "t_mean": float(np.mean([r["t_train"] for r in base_recs])),
            }
            if baseline_cache is not None:
                baseline_cache[cache_key] = base_stats
                save_baseline_cache(args.baseline_cache, baseline_cache, args)

        if args.baseline_only:
            continue

        if args.warmup > 0 and ratios:
            print(f"\nWarmup: model={model_name} ratio={ratios[0]} "
                  f"({args.warmup} run(s), discarded)")
            for warm_idx in range(args.warmup):
                warm_seed = int(args.base_seed) + 1_000_000 + warm_idx
                print(f"  warmup {warm_idx + 1}/{args.warmup}")
                run_methods_once(
                    mods, data, target_type, model_name, ratios[0],
                    method_keys, args, max_candidates, warm_seed)

        sweeps_by_method: Dict[str, List[Dict[str, Any]]] = {
            key: [] for key in method_keys
        }
        for ratio in ratios:
            print("\n" + "=" * 72)
            print(f"Ratio {ratio:.3f} ({1.0 / ratio:.2f}x target) "
                  f"model={model_name} max_candidates={max_candidates}")
            print("=" * 72)
            records_by_method: Dict[str, List[Dict[str, Any]]] = {
                key: [] for key in method_keys
            }
            for run_idx in range(args.runs):
                run_seed = int(args.base_seed) + run_idx
                print(f"\nRun {run_idx + 1}/{args.runs} (seed={run_seed})")
                recs = run_methods_once(
                    mods, data, target_type, model_name, ratio,
                    method_keys, args, max_candidates, run_seed)
                for rec in recs:
                    key = rec["method_key"]
                    records_by_method[key].append(rec)
                    row = dict(rec)
                    row.update({
                        "dataset": dataset,
                        "model": model_name,
                        "run": run_idx,
                        "max_candidates": int(max_candidates),
                        "variant_dir": str(variant_dir),
                        "dataset_total_nodes": dataset_info["total_nodes"],
                        "dataset_total_directed_edge_entries": dataset_info[
                            "total_directed_edge_entries"],
                        "dataset_target_type": dataset_info["target_type"],
                        "dataset_target_nodes": dataset_info["target_nodes"],
                        "dataset_target_num_classes": dataset_info[
                            "target_num_classes"],
                        "dataset_node_counts_json": json.dumps(
                            dataset_info["node_counts"], sort_keys=True),
                        "dataset_edge_counts_json": json.dumps(
                            dataset_info["edge_counts"], sort_keys=True),
                        "dataset_feature_dims_json": json.dumps(
                            dataset_info["feature_dims"], sort_keys=True),
                        "dataset_target_split_counts_json": json.dumps(
                            dataset_info["target_split_counts"],
                            sort_keys=True),
                    })
                    if base_stats is not None:
                        row.update({
                            "baseline_acc_mean": base_stats["acc_mean"],
                            "baseline_acc_std": base_stats["acc_std"],
                            "baseline_macro_f1_mean": base_stats[
                                "macro_f1_mean"],
                            "baseline_macro_f1_std": base_stats[
                                "macro_f1_std"],
                            "baseline_micro_f1_mean": base_stats[
                                "micro_f1_mean"],
                            "baseline_micro_f1_std": base_stats[
                                "micro_f1_std"],
                            "baseline_t_train": base_stats["t_mean"],
                        })
                    rows.append(row)
                    print(
                        f"  {rec['compressor']:<14} "
                        f"comp={rec['compression']:.2f}x "
                        f"acc={rec['test_acc']:.4f} "
                        f"t_comp={rec['t_compress']:.1f}s "
                        f"t_emb={rec['embedding_time']:.1f}s "
                        f"t_train={rec['t_train_wall']:.1f}s "
                        f"gpu_peak={rec['gpu_peak_allocated_total_bytes']}"
                    )
            for key, method_records in records_by_method.items():
                sweeps_by_method[key].append(
                    aggregate_ratio(method_records, ratio, base_stats))
            _write_table(
                variant_dir, f"{dataset}_rows_checkpoint", rows)

        for key in method_keys:
            label = display_method(key, method_keys)
            sweep = sweeps_by_method[key]
            experiments.print_sweep_table(base_stats, sweep, dataset)
            experiments.save_sweep_plot(
                sweep,
                dataset,
                model_name,
                str(variant_dir),
                label,
                args.ratio_search,
                args.merge_objective,
            )
            all_results[(model_name, label)] = (base_stats, sweep)

    if args.baseline_only:
        return rows

    display_labels = [display_method(key, method_keys) for key in method_keys]
    experiments.print_compressor_table(
        all_results, dataset, list(args.models), display_labels)
    experiments.save_comparison_outputs(
        all_results,
        dataset,
        list(args.models),
        display_labels,
        str(variant_dir),
        args.ratio_search,
        args.merge_objective,
    )
    return rows


def write_manifest(
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paper_pipeline_ablation_rows.csv"
    json_path = out_dir / "paper_pipeline_ablation_rows.json"
    meta_path = out_dir / "paper_pipeline_ablation_manifest.json"

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(list(rows)), f, indent=2)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(metadata), f, indent=2)

    print(f"[write] {csv_path}")
    print(f"[write] {json_path}")
    print(f"[write] {meta_path}")


def _write_table(out_dir: Path, stem: str,
                 rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[skip] no rows for {stem}")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(list(rows)), f, indent=2)
    print(f"[write] {csv_path}")
    print(f"[write] {json_path}")


def _paper_default_kind(row: Dict[str, Any]) -> Optional[str]:
    method_key = str(row.get("method_key", ""))
    try:
        max_candidates = int(row.get("max_candidates", -1))
    except (TypeError, ValueError):
        max_candidates = -1

    if (
        method_key == "hcgc_emb:gnn"
        and max_candidates == PAPER_HCGC_MAX_CANDIDATES
    ):
        return "HCGC@128"
    if (
        method_key == "cgc_homo_raw"
        and max_candidates == PAPER_CGC_HOMO_MAX_CANDIDATES
    ):
        return "CGC-Homo@128"
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std_rows(rows: Sequence[Dict[str, Any]],
                   key: str) -> Tuple[float, float]:
    vals = np.array([_safe_float(r.get(key)) for r in rows], dtype=float)
    if vals.size == 0 or np.isnan(vals).all():
        return float("nan"), float("nan")
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def _metric_key(rows: Sequence[Dict[str, Any]],
                raw_key: str, aggregate_key: str) -> str:
    return raw_key if any(raw_key in r for r in rows) else aggregate_key


def _aggregate_group(rows: Sequence[Dict[str, Any]],
                     extra: Dict[str, Any]) -> Dict[str, Any]:
    acc_m, acc_s = _mean_std_rows(
        rows, _metric_key(rows, "test_acc", "test_acc_mean"))
    mf1_m, mf1_s = _mean_std_rows(
        rows, _metric_key(rows, "test_macro_f1", "test_macro_f1_mean"))
    mif1_m, mif1_s = _mean_std_rows(
        rows, _metric_key(rows, "test_micro_f1", "test_micro_f1_mean"))
    comp_m, comp_s = _mean_std_rows(
        rows, _metric_key(rows, "compression", "compression_mean"))
    tc_m, tc_s = _mean_std_rows(
        rows, _metric_key(rows, "t_compress", "t_compress_mean"))
    tr_m, tr_s = _mean_std_rows(
        rows, _metric_key(rows, "t_train_wall", "t_train_mean"))
    storage_m, storage_s = _mean_std_rows(
        rows, _metric_key(rows, "storage_compression",
                          "storage_compression_mean"))
    ratio_err_m, ratio_err_s = _mean_std_rows(
        rows, _metric_key(rows, "retention_ratio_abs_error",
                          "retention_abs_error_mean"))
    gpu_m, gpu_s = _mean_std_rows(
        rows, _metric_key(rows, "gpu_peak_allocated_total_bytes",
                          "gpu_peak_allocated_bytes_mean"))
    b_acc_m, b_acc_s = _mean_std_rows(rows, "baseline_acc_mean")
    b_mf1_m, b_mf1_s = _mean_std_rows(rows, "baseline_macro_f1_mean")

    out = dict(extra)
    out.update({
        "runs_or_points": len(rows),
        "compression_mean": comp_m,
        "compression_std": comp_s,
        "test_acc_mean": acc_m,
        "test_acc_std": acc_s,
        "test_macro_f1_mean": mf1_m,
        "test_macro_f1_std": mf1_s,
        "test_micro_f1_mean": mif1_m,
        "test_micro_f1_std": mif1_s,
        "baseline_acc_mean": b_acc_m,
        "baseline_acc_std": b_acc_s,
        "baseline_macro_f1_mean": b_mf1_m,
        "baseline_macro_f1_std": b_mf1_s,
        "acc_drop": acc_m - b_acc_m,
        "macro_f1_drop": mf1_m - b_mf1_m,
        "t_compress_mean": tc_m,
        "t_compress_std": tc_s,
        "t_train_mean": tr_m,
        "t_train_std": tr_s,
        "storage_compression_mean": storage_m,
        "storage_compression_std": storage_s,
        "retention_abs_error_mean": ratio_err_m,
        "retention_abs_error_std": ratio_err_s,
        "gpu_peak_allocated_bytes_mean": gpu_m,
        "gpu_peak_allocated_bytes_std": gpu_s,
    })
    return out


def write_paper_default_check(out_dir: Path,
                              rows: Sequence[Dict[str, Any]]) -> None:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        paper_kind = _paper_default_kind(row)
        if paper_kind is None:
            continue
        copied = dict(row)
        copied["paper_default"] = paper_kind
        selected.append(copied)

    if not selected:
        print("[skip] no paper-default sanity rows found")
        return

    by_model_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in selected:
        key = (
            row.get("dataset"),
            row.get("model"),
            row.get("paper_default"),
            row.get("compressor"),
            row.get("ratio"),
            row.get("max_candidates"),
        )
        by_model_groups.setdefault(key, []).append(row)

    by_model: List[Dict[str, Any]] = []
    for key, group in by_model_groups.items():
        dataset, model, paper_default, compressor, ratio, max_candidates = key
        by_model.append(_aggregate_group(group, {
            "dataset": dataset,
            "model": model,
            "paper_default": paper_default,
            "compressor": compressor,
            "ratio": _safe_float(ratio),
            "max_candidates": int(max_candidates),
        }))
    by_model.sort(key=lambda r: (
        str(r["dataset"]), str(r["model"]), str(r["paper_default"]),
        -float(r["ratio"])))

    mean_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in by_model:
        key = (
            row["dataset"],
            row["paper_default"],
            row["compressor"],
            row["ratio"],
            row["max_candidates"],
        )
        mean_groups.setdefault(key, []).append(row)

    means: List[Dict[str, Any]] = []
    for key, group in mean_groups.items():
        dataset, paper_default, compressor, ratio, max_candidates = key
        means.append(_aggregate_group(group, {
            "dataset": dataset,
            "paper_default": paper_default,
            "compressor": compressor,
            "ratio": _safe_float(ratio),
            "max_candidates": int(max_candidates),
        }))
    means.sort(key=lambda r: (
        str(r["dataset"]), str(r["paper_default"]), -float(r["ratio"])))

    _write_table(out_dir, "paper_default_check_by_model", by_model)
    _write_table(out_dir, "paper_default_check_mean", means)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="HCGC manuscript max-candidate / embedding runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hcgc-root", default=None,
                        help=("Path to the HCGC checkout. By default, the "
                              "runner also checks the current working directory."))
    parser.add_argument("--datasets", "--dataset", nargs="+", default=["acm"])
    parser.add_argument("--acm-variant", choices=["paper", "full"],
                        default="paper",
                        help=("How to interpret --datasets acm. `paper` maps "
                              "ACM to the paper-style acm3 loader used by "
                              "Fig. 3; `full` keeps the full ACM loader. "
                              "Use --datasets full_acm to force the full "
                              "loader for one dataset entry."))
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.5, 0.3, 0.25, 0.2, 0.15, 0.1])
    parser.add_argument("--compressors", nargs="+",
                        default=["hcgc", "cgc_homo"])
    parser.add_argument("--models", nargs="+",
                        default=["sage", "rgcn", "gat", "appnp"])
    parser.add_argument("--max-candidates", nargs="+", type=int,
                        default=[128])
    parser.add_argument("--emb-methods", nargs="+",
                        choices=EMB_METHOD_CHOICES,
                        default=["gnn"],
                        help=("Embedding/coarsening representations to compare. "
                              "`raw` means original --no-pretrain; non-raw "
                              "values share one pretrain context between "
                              "HCGC-Emb and CGC-Homo-Emb."))

    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42,
                        help=("Timed run i uses base_seed + i. The same seed "
                              "is paired across ratios, models, and methods."))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--root", default=None,
                        help="Dataset root. Default: <hcgc-root>/data.")
    parser.add_argument("--plot-dir", default="results/paper_pipeline_ablation")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--baseline-cache", default=None,
                        help=("Optional JSON file for saving/reusing original "
                              "graph baseline metrics. Entries are reused only "
                              "when runs/train settings match."))
    parser.add_argument("--baseline-only", action="store_true",
                        help=("Train/cache original-graph baselines and exit "
                              "without running compression. Intended for very "
                              "large datasets such as ogbn-mag."))

    parser.add_argument("--ratio-search", default="fast",
                        choices=["fast", "precise"])
    parser.add_argument("--auto-search-runs", type=int, default=8)
    parser.add_argument("--auto-target-tolerance", type=float, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--raw-no-l2", action="store_true")
    parser.add_argument("--relprop-hops", type=int, default=2)
    parser.add_argument("--relprop-outdim", type=int, default=128)
    parser.add_argument("--type-thresholds", action="store_true")
    parser.add_argument("--metapath-thresholds", action="store_true")
    parser.add_argument("--pairwise-merge", dest="pairwise_merge",
                        action="store_true", default=True)
    parser.add_argument("--ball-multi-merge", dest="pairwise_merge",
                        action="store_false")
    parser.add_argument("--merge-objective", default="ward",
                        choices=["ward", "quotient_de"])
    parser.add_argument("--skip-reassignment", action="store_true")
    parser.add_argument("--edge-weight-mode", default="binary",
                        choices=["binary", "count", "log_count", "density"])
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--train-hidden", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--max-hub-degree", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    hcgc_root = resolve_hcgc_root(args.hcgc_root)
    if args.root is None:
        args.root = str(hcgc_root / "data")
    out_dir = Path(args.plot_dir).expanduser().resolve()
    ratios = sorted([float(r) for r in args.ratios], reverse=True)
    methods = method_labels(args.compressors, args.emb_methods)
    datasets, dataset_aliases = resolve_datasets(
        args.datasets, args.acm_variant)

    mods = import_original_pipeline(hcgc_root)
    validate_datasets(datasets, mods["experiments"].LOADERS)
    hcgc_git = git_checkout_metadata(hcgc_root)
    runner_git = git_checkout_metadata(SCRIPT_DIR)
    environment = runtime_environment_metadata()
    started_at_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 72)
    print("HCGC manuscript experiment runner")
    print("=" * 72)
    print(f"HCGC root     : {hcgc_root}")
    print(f"data root     : {args.root}")
    print(f"out dir       : {out_dir}")
    print(f"datasets      : {datasets}")
    if dataset_aliases:
        print(f"aliases       : {dataset_aliases}")
    print(f"models        : {args.models}")
    print(f"methods       : {[display_method(m, methods) for m in methods]}")
    print(f"ratios        : {ratios}")
    print(f"max candidates: {args.max_candidates}")
    print(f"emb methods   : {args.emb_methods}")
    print(f"run seeds     : {args.base_seed} .. "
          f"{args.base_seed + max(args.runs - 1, 0)}")
    print(f"HCGC commit   : {hcgc_git['commit'] or 'unavailable'}")
    print(f"HCGC dirty    : {hcgc_git['dirty']}")
    print(f"runner commit : {runner_git['commit'] or 'unavailable'}")
    print(f"runner dirty  : {runner_git['dirty']}")
    print("=" * 72)

    all_rows: List[Dict[str, Any]] = []
    baseline_cache: Dict[Tuple[str, str], Dict[str, Any]] = load_baseline_cache(
        args.baseline_cache, args)
    max_candidate_values = (
        [args.max_candidates[0]] if args.baseline_only else args.max_candidates
    )

    for max_candidates in max_candidate_values:
        variant_dir = out_dir / f"maxcand_{int(max_candidates)}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        for dataset in datasets:
            rows = run_dataset_sweeps(
                mods,
                dataset,
                methods,
                ratios,
                args,
                max_candidates,
                variant_dir,
                baseline_cache,
            )
            all_rows.extend(rows)
            write_manifest(
                out_dir,
                all_rows,
                {
                    "hcgc_root": str(hcgc_root),
                    "hcgc_git_commit": hcgc_git["commit"],
                    "hcgc_git_dirty": hcgc_git["dirty"],
                    "runner_git_commit": runner_git["commit"],
                    "runner_git_dirty": runner_git["dirty"],
                    "experiment_started_at_utc": started_at_utc,
                    "manifest_updated_at_utc": datetime.now(
                        timezone.utc).isoformat(),
                    "runtime_environment": environment,
                    "arguments": vars(args),
                    "base_seed": int(args.base_seed),
                    "timed_run_seeds": [
                        int(args.base_seed) + i for i in range(args.runs)
                    ],
                    "warmup_seed_policy": (
                        "base_seed + 1,000,000 + warmup_index; discarded"
                    ),
                    "data_root": str(args.root),
                    "out_dir": str(out_dir),
                    "requested_datasets": args.datasets,
                    "datasets": datasets,
                    "dataset_aliases": dataset_aliases,
                    "acm_variant": args.acm_variant,
                    "models": args.models,
                    "methods": [display_method(m, methods) for m in methods],
                    "ratios": ratios,
                    "max_candidates": args.max_candidates,
                    "emb_methods": args.emb_methods,
                    "note": (
                        "Dataset loading, HCGC phases, downstream training, "
                        "evaluation, and comparison output writers come from "
                        "the selected HCGC checkout. HCGC-Emb and "
                        "CGC-Homo-Emb share one pretrain context per ratio/run; "
                        "reported compression times include that embedding time. "
                        "By default --datasets acm resolves to acm3, the "
                        "paper-style ACM split used for Fig. 3."
                    ),
                },
            )
    write_paper_default_check(out_dir, all_rows)
    save_baseline_cache(args.baseline_cache, baseline_cache, args)


if __name__ == "__main__":
    main()
