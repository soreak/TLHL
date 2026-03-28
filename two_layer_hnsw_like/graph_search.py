from __future__ import annotations

import heapq
from typing import List, Tuple

import numpy as np

from .distance import l2_sq


def search_layer(
    query: np.ndarray,
    entry_points: List[int],
    vectors: np.ndarray,
    adj: List[List[int]],
    ef: int,
) -> List[int]:
    """
    Approximate single-layer graph search, similar to HNSW base-layer search.

    Returns candidate ids sorted by increasing distance to the query.
    """
    n = len(adj)
    if n == 0:
        return []

    entry_points = [ep for ep in entry_points if ep is not None and 0 <= ep < n]
    if not entry_points:
        entry_points = [0]

    ef = max(1, min(ef, n))

    visited = set()
    candidate_heap: List[Tuple[float, int]] = []  # min-heap: (dist, id)
    top_heap: List[Tuple[float, int]] = []        # max-heap simulated by (-dist, id)

    for ep in entry_points:
        if ep in visited:
            continue
        visited.add(ep)
        d = l2_sq(query, vectors[ep])
        heapq.heappush(candidate_heap, (d, ep))
        heapq.heappush(top_heap, (-d, ep))

    while candidate_heap:
        curr_dist, curr_id = heapq.heappop(candidate_heap)
        worst_best = -top_heap[0][0]

        if curr_dist > worst_best and len(top_heap) >= ef:
            break

        for nb in adj[curr_id]:
            if nb in visited:
                continue
            visited.add(nb)

            d = l2_sq(query, vectors[nb])
            if len(top_heap) < ef or d < worst_best:
                heapq.heappush(candidate_heap, (d, nb))
                heapq.heappush(top_heap, (-d, nb))
                if len(top_heap) > ef:
                    heapq.heappop(top_heap)
                worst_best = -top_heap[0][0]

    return [
        node_id
        for _, node_id in sorted(
            [(-neg_d, node_id) for neg_d, node_id in top_heap],
            key=lambda x: x[0],
        )
    ]
