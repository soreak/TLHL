from __future__ import annotations

from typing import List

import numpy as np

from .heuristic import exact_sorted_ids, heuristic_select


def mutual_connect(
    new_id: int,
    selected: List[int],
    vectors: np.ndarray,
    adj: List[List[int]],
    new_degree_limit: int,
    old_degree_limit: int,
) -> None:
    """
    HNSW-style mutual connection.

    - The new node keeps at most `new_degree_limit` neighbors.
    - Existing nodes accept the new node if possible.
    - If an existing node overflows, its neighbor list is re-pruned using the
      same diversity heuristic, with capacity `old_degree_limit`.
    """
    if new_id < 0 or new_id >= len(adj):
        raise IndexError("new_id 越界")

    selected = [x for x in selected if x != new_id]
    if len(selected) > new_degree_limit:
        selected = heuristic_select(
            query=vectors[new_id],
            candidate_ids=selected,
            vectors=vectors,
            max_neighbors=new_degree_limit,
        )

    adj[new_id] = list(selected)

    for nb in selected:
        merged = list(adj[nb])
        if new_id not in merged:
            merged.append(new_id)

        if len(merged) <= old_degree_limit:
            adj[nb] = merged
        else:
            merged = [x for x in merged if x != nb]
            merged = exact_sorted_ids(vectors[nb], vectors, merged)
            pruned = heuristic_select(
                query=vectors[nb],
                candidate_ids=merged,
                vectors=vectors,
                max_neighbors=old_degree_limit,
            )
            adj[nb] = pruned
