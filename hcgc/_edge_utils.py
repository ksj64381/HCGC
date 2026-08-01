"""Utilities for building the simple undirected graph used by HCGC."""

import numpy as np


def flatten_undirected_edge_groups(edge_groups, offsets, num_nodes):
    """Convert typed edge groups to unique global undirected node pairs.

    The C++ kernel expands every input pair into both CSR directions. PyG
    datasets commonly store reciprocal typed relations already, so passing
    those entries through unchanged would multiply the same adjacency edge.

    Args:
        edge_groups: Iterable of ``(source_type, destination_type, edge_index)``
            tuples. ``edge_index`` must have shape ``[2, num_edges]``.
        offsets: Mapping from node type to its global node-ID offset.
        num_nodes: Total number of nodes across all types.

    Returns:
        ``(src, dst, stats)`` where ``src`` and ``dst`` contain one canonical
        pair (``src < dst``) for each unique undirected edge.
    """
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")

    src_parts = []
    dst_parts = []
    input_entries = 0

    for source_type, destination_type, edge_index in edge_groups:
        if source_type not in offsets or destination_type not in offsets:
            continue

        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")

        source = edge_index[0].astype(np.int64, copy=False)
        destination = edge_index[1].astype(np.int64, copy=False)
        input_entries += int(source.size)
        if source.size == 0:
            continue

        src_parts.append(source + int(offsets[source_type]))
        dst_parts.append(destination + int(offsets[destination_type]))

    if src_parts:
        source = np.concatenate(src_parts)
        destination = np.concatenate(dst_parts)
    else:
        source = np.empty(0, dtype=np.int64)
        destination = np.empty(0, dtype=np.int64)

    if source.size and (
        source.min() < 0
        or destination.min() < 0
        or source.max() >= num_nodes
        or destination.max() >= num_nodes
    ):
        raise ValueError("edge_index contains a node outside the configured type ranges")

    non_self = source != destination
    self_loops_removed = int(source.size - np.count_nonzero(non_self))
    source = source[non_self]
    destination = destination[non_self]

    if source.size:
        lower = np.minimum(source, destination)
        upper = np.maximum(source, destination)
        keys = lower * num_nodes + upper
        unique_keys = np.unique(keys)
        source = unique_keys // num_nodes
        destination = unique_keys % num_nodes
    else:
        source = np.empty(0, dtype=np.int64)
        destination = np.empty(0, dtype=np.int64)

    unique_edges = int(source.size)
    non_self_entries = input_entries - self_loops_removed
    stats = {
        "input_edge_entries": input_entries,
        "self_loops_removed": self_loops_removed,
        "duplicate_entries_removed": non_self_entries - unique_edges,
        "unique_undirected_edges": unique_edges,
    }
    return (
        source.astype(np.int32, copy=False),
        destination.astype(np.int32, copy=False),
        stats,
    )
