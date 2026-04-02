from __future__ import annotations

import heapq
from typing import List, Tuple
import numpy as np
from .distance import l2_sq_fast, prepare_query_vector

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

    # 只做一次 query 预处理，避免热路径反复 np.asarray
    q = prepare_query_vector(query)

    visited = set()
    candidate_heap: List[Tuple[float, int]] = []  # min-heap: (dist, id)
    top_heap: List[Tuple[float, int]] = []        # max-heap simulated by (-dist, id)

    heappush = heapq.heappush
    heappop = heapq.heappop
    dist_fn = l2_sq_fast
    vecs = vectors
    graph = adj

    for ep in entry_points:
        if ep in visited:
            continue
        visited.add(ep)
        d = dist_fn(q, vecs[ep])
        heappush(candidate_heap, (d, ep))
        heappush(top_heap, (-d, ep))

    while candidate_heap:
        curr_dist, curr_id = heappop(candidate_heap)
        worst_best = -top_heap[0][0]

        if curr_dist > worst_best and len(top_heap) >= ef:
            break

        for nb in graph[curr_id]:
            if nb in visited:
                continue
            visited.add(nb)

            d = dist_fn(q, vecs[nb])
            if len(top_heap) < ef or d < worst_best:
                heappush(candidate_heap, (d, nb))
                heappush(top_heap, (-d, nb))
                if len(top_heap) > ef:
                    heappop(top_heap)
                worst_best = -top_heap[0][0]

    result = [(-neg_d, node_id) for neg_d, node_id in top_heap]
    result.sort(key=lambda x: x[0])
    return [node_id for _, node_id in result]
