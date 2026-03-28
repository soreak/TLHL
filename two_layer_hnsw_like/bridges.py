from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .distance import l2_sq
from .graph_search import search_layer
from .heuristic import exact_sorted_ids, heuristic_select


def build_bridges(
    centers: np.ndarray,
    labels: np.ndarray,
    base_vectors: np.ndarray,
    base_adj: List[List[int]],
    base_entry: Optional[int],
    m: int,
    ef_construction: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build cross-layer connections.

    For each center, search the base layer and connect the center to `m`
    representative nearby base nodes.
    """
    K = len(centers)
    N = len(base_vectors)

    center_to_base: List[List[int]] = [[] for _ in range(K)]
    base_to_center: List[List[int]] = [[] for _ in range(N)]

    if K == 0 or N == 0:
        return center_to_base, base_to_center

    ef_bridge = max(ef_construction, m * 4)

    for cid in range(K):
        center_vec = centers[cid]
        members = np.where(labels == cid)[0]

        if len(members) > 0:
            best_member = min(members, key=lambda idx: l2_sq(center_vec, base_vectors[idx]))
            entry_points = [int(best_member)]
        else:
            entry_points = [base_entry] if base_entry is not None else [0]

        cand = search_layer(
            query=center_vec,
            entry_points=entry_points,
            vectors=base_vectors,
            adj=base_adj,
            ef=min(ef_bridge, N),
        )

        if not cand:
            cand = list(range(N))
            cand = exact_sorted_ids(center_vec, base_vectors, cand)

        selected = heuristic_select(
            query=center_vec,
            candidate_ids=cand,
            vectors=base_vectors,
            max_neighbors=m,
        )

        if not selected:
            selected = cand[:min(m, len(cand))]

        center_to_base[cid] = selected
        for bid in selected:
            base_to_center[bid].append(cid)

    return center_to_base, base_to_center
