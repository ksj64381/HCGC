import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "hcgc" / "_edge_utils.py"
SPEC = importlib.util.spec_from_file_location("hcgc_edge_utils", MODULE_PATH)
EDGE_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EDGE_UTILS)


class FlattenUndirectedEdgeGroupsTest(unittest.TestCase):
    def test_reciprocal_relations_and_parallel_entries_are_deduplicated(self):
        groups = [
            ("paper", "author", np.array([[0, 0, 1], [0, 0, 1]])),
            ("author", "paper", np.array([[0, 0, 1], [0, 0, 1]])),
        ]

        src, dst, stats = EDGE_UTILS.flatten_undirected_edge_groups(
            groups, {"paper": 0, "author": 2}, num_nodes=4)

        np.testing.assert_array_equal(src, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(dst, np.array([2, 3], dtype=np.int32))
        self.assertEqual(stats["input_edge_entries"], 6)
        self.assertEqual(stats["duplicate_entries_removed"], 4)
        self.assertEqual(stats["unique_undirected_edges"], 2)

    def test_self_loops_are_removed_and_pairs_are_canonical(self):
        groups = [
            ("node", "node", np.array([[2, 3, 1, 0], [2, 1, 3, 1]])),
        ]

        src, dst, stats = EDGE_UTILS.flatten_undirected_edge_groups(
            groups, {"node": 0}, num_nodes=4)

        np.testing.assert_array_equal(src, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(dst, np.array([1, 3], dtype=np.int32))
        self.assertEqual(stats["self_loops_removed"], 1)
        self.assertEqual(stats["duplicate_entries_removed"], 1)

    def test_empty_input_returns_empty_int32_arrays(self):
        src, dst, stats = EDGE_UTILS.flatten_undirected_edge_groups(
            [], {"node": 0}, num_nodes=3)

        self.assertEqual(src.dtype, np.int32)
        self.assertEqual(dst.dtype, np.int32)
        self.assertEqual(src.size, 0)
        self.assertEqual(stats["unique_undirected_edges"], 0)

    def test_out_of_range_node_is_rejected(self):
        groups = [("node", "node", np.array([[0], [3]]))]
        with self.assertRaisesRegex(ValueError, "outside"):
            EDGE_UTILS.flatten_undirected_edge_groups(
                groups, {"node": 0}, num_nodes=3)


if __name__ == "__main__":
    unittest.main()
