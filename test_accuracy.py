from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    from . import TwoLayerHNSWLikeIndex
except ImportError:  # pragma: no cover
    from two_layer_hnsw_like import TwoLayerHNSWLikeIndex


@dataclass
class EvalResult:
    recall_at_k: float
    mean_overlap: float
    exact_mean_distance: float
    approx_mean_distance: float
    distance_ratio: float


def exact_topk(
    base_vectors: np.ndarray,
    query: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """Brute-force exact kNN using squared L2 distance."""
    dists = np.sum((base_vectors - query) ** 2, axis=1)
    topk_idx = np.argsort(dists)[:k]
    return [(int(i), float(dists[i])) for i in topk_idx]



def evaluate_recall(
    index: TwoLayerHNSWLikeIndex,
    base_vectors: np.ndarray,
    queries: np.ndarray,
    k: int,
    ef: int,
    n_probe_centers: int,
) -> EvalResult:
    """
    Measure ANN search quality against brute-force exact search.

    Metrics:
    - recall@k: exact top-k 被命中的比例
    - mean_overlap: 每个 query 的 top-k 交集比例均值
    - distance_ratio: 近似结果均值距离 / 精确结果均值距离
    """
    if k <= 0:
        raise ValueError("k 必须 > 0")
    if len(queries) == 0:
        raise ValueError("queries 不能为空")

    total_hits = 0
    total_possible = len(queries) * k
    overlap_ratios: List[float] = []
    exact_mean_dists: List[float] = []
    approx_mean_dists: List[float] = []

    for q in queries:
        exact = exact_topk(base_vectors, q, k)
        approx = index.search(q, k=k, ef=ef, n_probe_centers=n_probe_centers)

        exact_ids = [idx for idx, _ in exact]
        approx_ids = [idx for idx, _ in approx]
        exact_set = set(exact_ids)
        approx_set = set(approx_ids)

        hits = len(exact_set & approx_set)
        total_hits += hits
        overlap_ratios.append(hits / k)

        exact_mean_dists.append(float(np.mean([dist for _, dist in exact])))
        approx_mean_dists.append(float(np.mean([dist for _, dist in approx])))

    recall_at_k = total_hits / total_possible
    mean_overlap = float(np.mean(overlap_ratios))
    exact_mean_distance = float(np.mean(exact_mean_dists))
    approx_mean_distance = float(np.mean(approx_mean_dists))
    distance_ratio = (
        approx_mean_distance / exact_mean_distance
        if exact_mean_distance > 0
        else 1.0
    )

    return EvalResult(
        recall_at_k=recall_at_k,
        mean_overlap=mean_overlap,
        exact_mean_distance=exact_mean_distance,
        approx_mean_distance=approx_mean_distance,
        distance_ratio=distance_ratio,
    )



def make_synthetic_dataset(
    n_samples: int,
    dim: int,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    centers = rng.normal(loc=0.0, scale=8.0, size=(n_clusters, dim)).astype(np.float32)
    counts = np.full(n_clusters, n_samples // n_clusters, dtype=int)
    counts[: n_samples % n_clusters] += 1

    chunks = []
    for i, cnt in enumerate(counts):
        scale = rng.uniform(0.4, 1.2)
        chunk = rng.normal(loc=centers[i], scale=scale, size=(cnt, dim))
        chunks.append(chunk.astype(np.float32))

    X = np.vstack(chunks).astype(np.float32)
    rng.shuffle(X)
    return X



def sample_queries(
    base_vectors: np.ndarray,
    n_queries: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(base_vectors), size=n_queries, replace=False)
    queries = base_vectors[idx].copy()
    if noise_std > 0:
        queries += rng.normal(0.0, noise_std, size=queries.shape).astype(np.float32)
    return queries.astype(np.float32)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recall of TwoLayerHNSWLikeIndex")
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--n-centers", type=int, default=32)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--n-probe-centers", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    if args.n_queries > args.n_samples:
        raise ValueError("n_queries 不能大于 n_samples")

    X = make_synthetic_dataset(
        n_samples=args.n_samples,
        dim=args.dim,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    queries = sample_queries(
        base_vectors=X,
        n_queries=args.n_queries,
        noise_std=args.noise_std,
        seed=args.seed + 1,
    )

    index = TwoLayerHNSWLikeIndex(
        n_centers=args.n_centers,
        m=args.m,
        ef_construction=args.ef_construction,
        random_state=args.seed,
    ).fit(X)

    print("index summary:")
    for key, value in index.summary().items():
        print(f"  {key}: {value}")

    print("\naccuracy evaluation:")
    print(
        f"  dataset={args.n_samples}x{args.dim}, queries={args.n_queries}, "
        f"k={args.k}, n_probe_centers={args.n_probe_centers}"
    )
    print(
        "  {:>10} | {:>10} | {:>12} | {:>16}".format(
            "ef_search", "recall@k", "mean_overlap", "distance_ratio"
        )
    )
    print("  " + "-" * 59)

    for ef in args.ef_search:
        result = evaluate_recall(
            index=index,
            base_vectors=X,
            queries=queries,
            k=args.k,
            ef=ef,
            n_probe_centers=args.n_probe_centers,
        )
        print(
            "  {:>10} | {:>10.4f} | {:>12.4f} | {:>16.4f}".format(
                ef,
                result.recall_at_k,
                result.mean_overlap,
                result.distance_ratio,
            )
        )

    print("\ninterpretation:")
    print("  - recall@k 越接近 1 越好")
    print("  - mean_overlap 越接近 1 越好")
    print("  - distance_ratio 越接近 1 越好；大于 1 表示近似结果平均距离更差")


if __name__ == "__main__":
    main()
