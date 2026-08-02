import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]


def load_module(name, path, injected_modules):
    previous = {key: sys.modules.get(key) for key in injected_modules}
    sys.modules.update(injected_modules)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class _NodeStore:
    num_nodes = 4


class _EdgeIndex:
    shape = (2, 3)


class _EdgeStore:
    edge_index = _EdgeIndex()


class _Data:
    node_types = ["node"]
    edge_types = [("node", "to", "node")]

    def __getitem__(self, key):
        return _NodeStore() if key == "node" else _EdgeStore()


def result_record(**kwargs):
    return {
        "compression": 2.0,
        "edge_compression": 1.5,
        "storage_compression": 1.4,
        "t_total": 1.0,
        "t_compress": 0.4,
        "t_train": 0.6,
        "test_acc": 0.75,
        "test_macro_f1": 0.74,
        "test_micro_f1": 0.75,
        **kwargs,
    }


class AblationProtocolTest(unittest.TestCase):
    def test_run_sweep_uses_distinct_timed_and_discarded_warmup_seeds(self):
        calls = []

        def run_once(*args, **kwargs):
            calls.append(dict(kwargs))
            return result_record(
                seed=kwargs["run_seed"],
                max_candidates=kwargs["max_candidates"],
            )

        benchmark = types.ModuleType("benchmark")
        benchmark.LOADERS = {"imdb": lambda root: (_Data(), "node")}
        benchmark._COMPRESSORS = {"hcgc": "HCGC"}
        benchmark._DOWNSTREAM_MODELS = {"sage": object()}
        benchmark._add_degree_features = lambda data: data
        benchmark._FULL_BATCH_NODE_LIMIT = 100_000
        benchmark.run_baseline = lambda *args, **kwargs: None
        benchmark.run_once = run_once

        experiments = load_module(
            "experiments_seed_test",
            ROOT / "experiments.py",
            {"benchmark": benchmark},
        )
        _, sweep = experiments.run_sweep(
            dataset="imdb",
            ratios=[0.5],
            runs=2,
            warmup=1,
            baseline=False,
            max_candidates=128,
            base_seed=42,
            pretrain_epochs=100,
            pretrain_patience=5,
        )

        self.assertEqual(
            [call["run_seed"] for call in calls],
            [1_000_042, 1_000_042, 42, 43],
        )
        self.assertTrue(all(call["max_candidates"] == 128 for call in calls))
        self.assertEqual(sweep[0]["run_seeds"], [42, 43])
        self.assertEqual(
            [row["seed"] for row in sweep[0]["run_records"]], [42, 43])

    def test_ablation_defaults_match_the_paper_protocol(self):
        benchmark = types.ModuleType("benchmark")
        benchmark.LOADERS = {
            "imdb": object(),
            "dblp": object(),
            "acm3": object(),
        }
        benchmark._DOWNSTREAM_MODELS = {
            "sage": object(),
            "rgcn": object(),
            "gat": object(),
            "appnp": object(),
        }
        experiments = types.ModuleType("experiments")
        experiments.run_sweep = lambda **kwargs: (None, [])

        module = load_module(
            "ablation_defaults_test",
            ROOT / "ablation_experiments.py",
            {"benchmark": benchmark, "experiments": experiments},
        )
        stdout = io.StringIO()
        with mock.patch.object(
            sys, "argv", ["ablation_experiments.py", "--dry-run"]
        ), contextlib.redirect_stdout(stdout):
            module.main()

        output = stdout.getvalue()
        self.assertIn("datasets : ['imdb', 'dblp']", output)
        self.assertIn("models   : ['sage', 'rgcn', 'gat', 'appnp']", output)
        self.assertIn(
            "variants : ['full', 'no_embedding', 'ball_multi', 'no_reassign']",
            output,
        )
        self.assertIn("seeds    : 42 .. 51", output)
        self.assertIn("max cand : 128", output)
        self.assertIn("search   : precise, runs=12, tolerance=0.02", output)


if __name__ == "__main__":
    unittest.main()
