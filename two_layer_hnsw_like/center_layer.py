from __future__ import annotations

from typing import List

import numpy as np

from .heuristic import exact_sorted_ids, heuristic_select


def build_center_layer(
    centers: np.ndarray,
    m: int,
    max_degree: int,
) -> List[List[int]]:
    """
    Build a sparse graph over KMeans centers.

    Each center first selects up to `m` outgoing neighbors using exact distances
    and the HNSW-style heuristic. Then reverse connections are added, and if an
    existing node exceeds `max_degree`, its adjacency list is re-pruned.
    """
    K = len(centers)
    adj: List[List[int]] = [[] for _ in range(K)]

    if K <= 1:
        return adj

    raw_out: List[List[int]] = [[] for _ in range(K)]
    for cid in range(K):
        cand = [j for j in range(K) if j != cid]
        cand = exact_sorted_ids(centers[cid], centers, cand)
        raw_out[cid] = heuristic_select(
            query=centers[cid],
            candidate_ids=cand,
            vectors=centers,
            max_neighbors=m,
        )

    adj = [list(nei) for nei in raw_out]

    for cid in range(K):
        for nb in raw_out[cid]:
            merged = list(adj[nb])
            if cid not in merged:
                merged.append(cid)

            if len(merged) <= max_degree:
                adj[nb] = merged
            else:
                merged = [x for x in merged if x != nb]
                merged = exact_sorted_ids(centers[nb], centers, merged)
                adj[nb] = heuristic_select(
                    query=centers[nb],
                    candidate_ids=merged,
                    vectors=centers,
                    max_neighbors=max_degree,
                )

    cleaned_adj: List[List[int]] = []
    for cid in range(K):
        seen = set()
        cleaned: List[int] = []
        for nb in adj[cid]:
            if nb == cid or nb in seen:
                continue
            seen.add(nb)
            cleaned.append(nb)

        if len(cleaned) > max_degree:
            cleaned = exact_sorted_ids(centers[cid], centers, cleaned)
            cleaned = heuristic_select(
                query=centers[cid],
                candidate_ids=cleaned,
                vectors=centers,
                max_neighbors=max_degree,
            )
        cleaned_adj.append(cleaned)

    return cleaned_adj
