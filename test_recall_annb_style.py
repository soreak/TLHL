from __future__ import annotations

import os
import time

import numpy as np
import pytest

from gt_top100_faiss import load_subset_by_gt, load_true_knn, save_true_knn_topk_faiss_hdf5
from two_layer_hnsw_like import TwoLayerHNSWLikeIndex


def annb_recall_from_distances(
    *,
    true_distances: np.ndarray,
    run_distances: np.ndarray,
    k_eval: int,
    epsilon: float,
) -> float:
    """ANN-Benchmarks style recall.

    A returned result counts as correct when its distance is not larger than the
    ground-truth k-th distance threshold.
    """
    thresholds = true_distances[:, k_eval - 1] + float(epsilon)
    actual = (run_distances[:, :k_eval] <= thresholds[:, None]).sum(axis=1)
    recalls = actual.astype(np.float32) / float(k_eval)
    return float(np.mean(recalls))



def overlap_recall(
    *,
    true_indices: np.ndarray,
    run_indices: np.ndarray,
    k_eval: int,
) -> float:
    recalls = np.empty(true_indices.shape[0], dtype=np.float32)
    for i in range(true_indices.shape[0]):
        gt = set(map(int, true_indices[i, :k_eval]))
        pred = set(map(int, run_indices[i, :k_eval]))
        recalls[i] = len(gt & pred) / float(k_eval)
    return float(np.mean(recalls))



def run_two_layer_index_queries(
    *,
    base: np.ndarray,
    queries: np.ndarray,
    k_eval: int,
    ef_search: int,
    n_centers: int,
    m: int,
    ef_construction: int,
    n_probe_centers: int,
    query_is_base_prefix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TwoLayerHNSWLikeIndex on the given base/query set."""
    t1 = time.perf_counter()
    index = TwoLayerHNSWLikeIndex(
        n_centers=n_centers,
        m=m,
        ef_construction=ef_construction,
        random_state=42,
    ).fit(base)
    t2 = time.perf_counter()

    run_indices = np.empty((queries.shape[0], k_eval), dtype=np.int32)
    run_distances = np.empty((queries.shape[0], k_eval), dtype=np.float32)

    internal_k = k_eval + 1 if query_is_base_prefix else k_eval

    for i, q in enumerate(queries):
        ans = index.search(
            q,
            k=internal_k,
            ef=ef_search,
            n_probe_centers=n_probe_centers,
        )
        ids = np.array([idx for idx, _ in ans], dtype=np.int32)
        dists = np.array([dist for _, dist in ans], dtype=np.float32)

        if query_is_base_prefix:
            mask = ids != i
            ids = ids[mask][:k_eval]
            dists = dists[mask][:k_eval]
        else:
            ids = ids[:k_eval]
            dists = dists[:k_eval]

        if ids.shape[0] < k_eval:
            raise ValueError(
                f"Query {i} returned only {ids.shape[0]} results after filtering, expected {k_eval}"
            )

        run_indices[i] = ids
        run_distances[i] = dists

    t3 = time.perf_counter()
    print(f"create={(t2 - t1) * 1000:.3f} ms query={(t3 - t2) * 1000:.3f} ms")
    return run_indices, run_distances



def test_two_layer_index_recall() -> None:
    hdf5_path = "./two_layer_hnsw_like/data/sift-128-euclidean.hdf5"
    mode = "self"               # 可选: "self" / "annb_test"
    base_limit = 100000          # 例如只用前 10w 个 train 点建库/生成真值
    query_limit = None           # annb_test 模式下可设成 10000 之类
    eval_first_n_queries = 100   # 本次测试只跑前 i 个 query；None 表示全量

    k_true = 100
    k_eval = 100
    epsilon = 1e-3
    gt_path = "gt_top100_faiss_first100k.npz"

    # ANN 参数
    n_centers = 4000
    m = 12
    ef_construction = 100
    ef_search = 10
    n_probe_centers = 1

    min_annb_recall = 0.95
    min_overlap_recall = 0.95

    if not os.path.exists(hdf5_path):
        pytest.skip(f"Dataset not found: {hdf5_path}")

    if k_true < k_eval:
        raise ValueError(f"k_true ({k_true}) must be >= k_eval ({k_eval})")

    # 1) 真值缓存
    if not os.path.exists(gt_path):
        print(f"create gt_path")
        save_true_knn_topk_faiss_hdf5(
            hdf5_path=hdf5_path,
            mode=mode,
            base_limit=base_limit,
            query_limit=query_limit,
            k_true=k_true,
            out_path=gt_path,
            metric="euclidean",
        )

    gt = load_true_knn(gt_path)
    base, queries = load_subset_by_gt(gt)

    # 只测试前 i 个 query
    total_queries = queries.shape[0]
    n_eval = total_queries if eval_first_n_queries is None else min(int(eval_first_n_queries), total_queries)
    if n_eval <= 0:
        raise ValueError(f"eval_first_n_queries must be > 0 or None, got {eval_first_n_queries}")

    queries_eval = queries[:n_eval]
    true_indices_top = gt["true_indices_base"][:n_eval, :k_eval]
    true_distances_top = gt["true_distances"][:n_eval, :k_eval]

    print(
        f"mode={gt['mode']} base={base.shape[0]} total_queries={total_queries} "
        f"eval_queries={n_eval} k_eval={k_eval} gt_backend={gt.get('backend')}"
    )

    # 2) 构图 + 查询
    run_indices, run_distances = run_two_layer_index_queries(
        base=base,
        queries=queries_eval,
        k_eval=k_eval,
        ef_search=ef_search,
        n_centers=n_centers,
        m=m,
        ef_construction=ef_construction,
        n_probe_centers=n_probe_centers,
        query_is_base_prefix=(str(gt["mode"]) == "self"),
    )

    # 3) accuracy
    annb_recall = annb_recall_from_distances(
        true_distances=true_distances_top,
        run_distances=run_distances,
        k_eval=k_eval,
        epsilon=epsilon,
    )
    overlap = overlap_recall(
        true_indices=true_indices_top,
        run_indices=run_indices,
        k_eval=k_eval,
    )

    print(f"ANNB recall@{k_eval}: {annb_recall:.4f}")
    print(f"Overlap recall@{k_eval}: {overlap:.4f}")

    assert annb_recall >= min_annb_recall
    assert overlap >= min_overlap_recall
