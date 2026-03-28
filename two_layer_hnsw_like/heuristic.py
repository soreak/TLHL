from __future__ import annotations

from typing import List

import numpy as np

from .distance import l2_sq


def exact_sorted_ids(query: np.ndarray, vectors: np.ndarray, ids: List[int]) -> List[int]:
    """Sort candidate ids by exact squared L2 distance to query."""
    return sorted(ids, key=lambda idx: l2_sq(query, vectors[idx]))


def heuristic_select(
    query: np.ndarray,
    candidate_ids: List[int],
    vectors: np.ndarray,
    max_neighbors: int,
) -> List[int]:
    """
    HNSW-style diversity heuristic used to choose neighbors.

    Candidates are first sorted by dist(query, cand). A candidate cand is rejected if
    there already exists a selected node sel such that:
        dist(sel, cand) < dist(query, cand)

    This tends to remove redundant neighbors from the same direction.
    """
    if max_neighbors <= 0:
        return []

    seen = set()
    ordered: List[int] = []
    for cid in candidate_ids:
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)

    ordered.sort(key=lambda cid: l2_sq(query, vectors[cid]))

    selected: List[int] = []
    for cid in ordered:
        d_q_c = l2_sq(query, vectors[cid])
        ok = True
        for sid in selected:
            d_s_c = l2_sq(vectors[sid], vectors[cid])
            if d_s_c < d_q_c:
                ok = False
                break
        if ok:
            selected.append(cid)
            if len(selected) >= max_neighbors:
                break

    # Optional fill-up to avoid creating graphs that are too sparse.
    if len(selected) < min(max_neighbors, len(ordered)):
        used = set(selected)
        for cid in ordered:
            if cid in used:
                continue
            selected.append(cid)
            used.add(cid)
            if len(selected) >= max_neighbors:
                break

    return selected
