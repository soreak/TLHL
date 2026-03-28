from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .graph_connect import mutual_connect
from .graph_search import search_layer
from .heuristic import exact_sorted_ids, heuristic_select


def build_base_layer(
    base_vectors: np.ndarray,
    m: int,
    ef_construction: int,
    max_degree: int,
    seed_adj: Optional[List[List[int]]] = None,
    seed_count: int = 0,
    insertion_entry_points: Optional[Sequence[int]] = None,
) -> Tuple[List[List[int]], Optional[int]]:
    """
    Build the base graph with an HNSW level-0 style incremental procedure.

    Parameters:
        base_vectors:
            All vectors that live in the base graph. In the modified design,
            the first `seed_count` nodes are virtual center nodes and the rest
            are real data nodes.
        seed_adj:
            Optional adjacency for the initial seed subgraph. It is typically the
            center-layer graph copied into the first `seed_count` base nodes.
        seed_count:
            Number of already-existing seed nodes at the front of `base_vectors`.
        insertion_entry_points:
            Optional entry point for each real data node insertion. When provided,
            item `i` corresponds to the node inserted at graph id `seed_count + i`.

    Returns:
        adj: adjacency list of the base layer
        entry: last inserted node used as an entry point for later search
    """
    N = len(base_vectors)
    adj: List[List[int]] = [[] for _ in range(N)]
    entry: Optional[int] = None

    if N == 0:
        return adj, entry

    if seed_count < 0 or seed_count > N:
        raise ValueError("seed_count 必须在 [0, N] 范围内")

    if seed_adj is not None:
        if len(seed_adj) != seed_count:
            raise ValueError("seed_adj 的长度必须等于 seed_count")
        for idx in range(seed_count):
            adj[idx] = list(seed_adj[idx])

    start_idx = 0
    if seed_count > 0:
        entry = seed_count - 1
        start_idx = seed_count
    else:
        entry = 0
        start_idx = 1

    for idx in range(start_idx, N):
        query = base_vectors[idx]

        entry_points: List[int] = []
        if insertion_entry_points is not None and idx >= seed_count:
            real_offset = idx - seed_count
            if 0 <= real_offset < len(insertion_entry_points):
                entry_points.append(int(insertion_entry_points[real_offset]))

        if entry is not None and entry not in entry_points:
            entry_points.append(entry)
        if not entry_points:
            entry_points = [0]

        effective_ef = min(ef_construction, max(1, idx))
        cand = search_layer(
            query=query,
            entry_points=entry_points,
            vectors=base_vectors,
            adj=adj,
            ef=effective_ef,
        )

        if not cand:
            cand = list(range(idx))
            cand = exact_sorted_ids(query, base_vectors, cand)

        selected = heuristic_select(
            query=query,
            candidate_ids=cand,
            vectors=base_vectors,
            max_neighbors=m,
        )

        if not selected:
            selected = cand[:1]

        mutual_connect(
            new_id=idx,
            selected=selected,
            vectors=base_vectors,
            adj=adj,
            new_degree_limit=m,
            old_degree_limit=max_degree,
        )

        entry = idx

    return adj, entry
