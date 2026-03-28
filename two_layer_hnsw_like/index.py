from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from .base_layer import build_base_layer
from .center_layer import build_center_layer
from .distance import l2_sq, to_float32_matrix
from .graph_search import search_layer


@dataclass
class IndexParams:
    n_centers: int
    m: int = 16
    ef_construction: int = 200
    center_max_degree: Optional[int] = None
    base_max_degree: Optional[int] = None
    random_state: int = 42

    # clustering
    cluster_method: str = "cpd_kmeans"
    cluster_max_iter: int = 20
    cluster_tol: float = 1e-6
    cluster_train_sample_size: Optional[int] = None
    cpd_psd_sample: int = 64
    cpd_psd_exact_k: int = 512
    cpd_mean_sample: int = 64
    cpd_mean_exact_k: int = 512

    def __post_init__(self) -> None:
        if self.n_centers <= 0:
            raise ValueError("n_centers 必须 > 0")
        if self.m <= 0:
            raise ValueError("m 必须 > 0")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction 必须 > 0")
        if self.cluster_max_iter <= 0:
            raise ValueError("cluster_max_iter 必须 > 0")
        if self.cluster_tol <= 0:
            raise ValueError("cluster_tol 必须 > 0")
        if self.center_max_degree is None:
            self.center_max_degree = 2 * self.m
        if self.base_max_degree is None:
            self.base_max_degree = 2 * self.m
        self.cluster_method = str(self.cluster_method).lower()
        if self.cluster_method not in {"cpd_kmeans", "kmeans"}:
            raise ValueError("cluster_method 必须是 'cpd_kmeans' 或 'kmeans'")


class TwoLayerHNSWLikeIndex:
    """
    Two-layer graph index with virtual center nodes seeded into the base graph.

    Layer 1: center layer
      - balanced clustering (default: CPD_KMeans) produces K virtual centers
      - centers are graph nodes, connected sparsely

    Layer 2: base graph layer
      - the first K base nodes are the same virtual centers
      - original data points are inserted after them using the HNSW level-0 style
      - query results must exclude the virtual center nodes
    """

    def __init__(
        self,
        n_centers: int,
        m: int = 16,
        ef_construction: int = 200,
        center_max_degree: Optional[int] = None,
        base_max_degree: Optional[int] = None,
        random_state: int = 42,
        cluster_method: str = "cpd_kmeans",
        cluster_max_iter: int = 20,
        cluster_tol: float = 1e-6,
        cluster_train_sample_size: Optional[int] = None,
        cpd_psd_sample: int = 64,
        cpd_psd_exact_k: int = 512,
        cpd_mean_sample: int = 64,
        cpd_mean_exact_k: int = 512,
    ):
        self.params = IndexParams(
            n_centers=n_centers,
            m=m,
            ef_construction=ef_construction,
            center_max_degree=center_max_degree,
            base_max_degree=base_max_degree,
            random_state=random_state,
            cluster_method=cluster_method,
            cluster_max_iter=cluster_max_iter,
            cluster_tol=cluster_tol,
            cluster_train_sample_size=cluster_train_sample_size,
            cpd_psd_sample=cpd_psd_sample,
            cpd_psd_exact_k=cpd_psd_exact_k,
            cpd_mean_sample=cpd_mean_sample,
            cpd_mean_exact_k=cpd_mean_exact_k,
        )

        self.data_vectors: Optional[np.ndarray] = None
        self.base_vectors: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

        self.center_adj: List[List[int]] = []
        self.base_adj: List[List[int]] = []

        self.base_entry: Optional[int] = None
        self.virtual_base_count: int = 0

    def _cluster_points(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster X into balanced virtual centers."""
        if self.params.cluster_method == "kmeans":
            kmeans = KMeans(
                n_clusters=self.params.n_centers,
                random_state=self.params.random_state,
                n_init=10,
                max_iter=self.params.cluster_max_iter,
                tol=self.params.cluster_tol,
            )
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
            return centers, labels.astype(np.int32, copy=False)

        # default: CPD_KMeans balanced clustering
        from .CPD_Kmeans_old import CPD_KMeans

        cpd = CPD_KMeans(
            data=X,
            k=self.params.n_centers,
            max_iter=self.params.cluster_max_iter,
            tol=self.params.cluster_tol,
            random_state=self.params.random_state,
            train_sample_size=self.params.cluster_train_sample_size,
            psd_sample=self.params.cpd_psd_sample,
            psd_exact_k=self.params.cpd_psd_exact_k,
            mean_sample=self.params.cpd_mean_sample,
            mean_exact_k=self.params.cpd_mean_exact_k,
        )
        centers, labels = cpd.train()
        centers = np.asarray(centers, dtype=np.float32, order="C")
        labels = np.asarray(labels, dtype=np.int32)
        if centers.shape[0] != self.params.n_centers:
            raise ValueError("CPD_KMeans 返回的 center 数量与 n_centers 不一致")
        if labels.shape[0] != X.shape[0]:
            raise ValueError("CPD_KMeans 返回的 labels 长度与样本数不一致")
        return centers, labels

    def fit(self, X: np.ndarray) -> "TwoLayerHNSWLikeIndex":
        X = to_float32_matrix(X)
        N = len(X)

        if self.params.n_centers > N:
            raise ValueError(
                f"n_centers={self.params.n_centers} 不能大于样本数 N={N}"
            )

        self.data_vectors = X

        self.centers, self.labels_ = self._cluster_points(X)

        self.center_adj = build_center_layer(
            centers=self.centers,
            m=self.params.m,
            max_degree=self.params.center_max_degree,
        )

        self.virtual_base_count = int(len(self.centers))
        self.base_vectors = np.ascontiguousarray(
            np.vstack([self.centers, self.data_vectors]).astype(np.float32)
        )

        insertion_entry_points = [int(label) for label in self.labels_]
        self.base_adj, self.base_entry = build_base_layer(
            base_vectors=self.base_vectors,
            m=self.params.m,
            ef_construction=self.params.ef_construction,
            max_degree=self.params.base_max_degree,
            seed_adj=self.center_adj,
            seed_count=self.virtual_base_count,
            insertion_entry_points=insertion_entry_points,
        )

        return self

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        n_probe_centers: int = 1,
    ) -> List[Tuple[int, float]]:
        if self.base_vectors is None or self.centers is None or self.data_vectors is None:
            raise RuntimeError("请先 fit()")

        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.base_vectors.shape[1]:
            raise ValueError("query 维度与训练数据不一致")

        if ef is None:
            ef = max(self.params.ef_construction, k * 4)

        center_dists = np.sum((self.centers - query) ** 2, axis=1)
        probe_count = max(1, min(n_probe_centers, len(self.centers)))
        probe_ids = np.argsort(center_dists)[:probe_count]

        entry_points: List[int] = [int(cid) for cid in probe_ids]
        if self.base_entry is not None and self.base_entry not in entry_points:
            entry_points.append(self.base_entry)

        ef_internal = min(
            max(ef, k) + self.virtual_base_count,
            len(self.base_vectors),
        )
        cand = search_layer(
            query=query,
            entry_points=entry_points,
            vectors=self.base_vectors,
            adj=self.base_adj,
            ef=ef_internal,
        )

        if not cand:
            cand = list(range(self.virtual_base_count, len(self.base_vectors)))

        filtered = [idx for idx in cand if idx >= self.virtual_base_count]
        result = [
            (idx - self.virtual_base_count, l2_sq(query, self.base_vectors[idx]))
            for idx in filtered
        ]
        result.sort(key=lambda x: x[1])
        return result[:k]

    def summary(self) -> dict:
        graph_n = 0 if self.base_vectors is None else int(len(self.base_vectors))
        real_n = 0 if self.data_vectors is None else int(len(self.data_vectors))
        virtual_n = int(self.virtual_base_count)

        avg_virtual_degree = 0.0
        avg_real_degree = 0.0
        if self.base_adj:
            if virtual_n > 0:
                avg_virtual_degree = float(np.mean([len(x) for x in self.base_adj[:virtual_n]]))
            if graph_n > virtual_n:
                avg_real_degree = float(np.mean([len(x) for x in self.base_adj[virtual_n:]]))

        cluster_hist = None
        if self.labels_ is not None and len(self.labels_) > 0:
            counts = np.bincount(self.labels_, minlength=self.params.n_centers)
            cluster_hist = {
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": float(counts.mean()),
                "std": float(counts.std()),
            }

        return {
            "cluster_method": self.params.cluster_method,
            "n_real_base_nodes": real_n,
            "n_virtual_base_nodes": virtual_n,
            "n_graph_nodes": graph_n,
            "n_centers": 0 if self.centers is None else int(len(self.centers)),
            "m": self.params.m,
            "ef_construction": self.params.ef_construction,
            "center_max_degree": self.params.center_max_degree,
            "base_max_degree": self.params.base_max_degree,
            "cluster_balance": cluster_hist,
            "avg_center_degree": float(np.mean([len(x) for x in self.center_adj])) if self.center_adj else 0.0,
            "avg_virtual_base_degree": avg_virtual_degree,
            "avg_real_base_degree": avg_real_degree,
        }
