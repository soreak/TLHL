from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np


def _load_subset_from_hdf5(
    hdf5_path: str | Path,
    *,
    mode: Literal["self", "annb_test"] = "self",
    base_limit: int | None = None,
    query_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a front subset from an ANN-Benchmarks HDF5 dataset.

    self:
        base = train[:base_limit], queries = same subset
    annb_test:
        base = train[:base_limit], queries = test[:query_limit]
    """
    with h5py.File(hdf5_path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        n_base = train.shape[0] if base_limit is None else min(int(base_limit), train.shape[0])
        if n_base <= 0:
            raise ValueError(f"base_limit must be > 0, got {base_limit}")
        base = np.ascontiguousarray(train[:n_base], dtype=np.float32)

        if mode == "self":
            return base, base

        test = np.asarray(f["test"], dtype=np.float32)
        n_query = test.shape[0] if query_limit is None else min(int(query_limit), test.shape[0])
        if n_query <= 0:
            raise ValueError(f"query_limit must be > 0, got {query_limit}")
        queries = np.ascontiguousarray(test[:n_query], dtype=np.float32)
        return base, queries



def save_true_knn_topk_faiss_hdf5(
    *,
    hdf5_path: str | Path,
    mode: Literal["self", "annb_test"] = "self",
    base_limit: int | None = None,
    query_limit: int | None = None,
    k_true: int = 100,
    out_path: str | Path = "gt_top100_faiss.npz",
    metric: str = "euclidean",
) -> str:
    """Build exact top-k ground truth from an ANN-Benchmarks HDF5 file.

    The output is a compressed NPZ cache, in the same spirit as the example you
    provided, but extended with distances and subset metadata.
    """
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FAISS is required for save_true_knn_topk_faiss_hdf5") from exc

    base, queries = _load_subset_from_hdf5(
        hdf5_path,
        mode=mode,
        base_limit=base_limit,
        query_limit=query_limit,
    )

    if metric in ("cosine", "dot"):
        def _l2norm_inplace(a: np.ndarray) -> None:
            norms = np.sqrt((a * a).sum(axis=1, keepdims=True))
            norms[norms == 0] = 1.0
            a /= norms

        base = base.copy(order="C")
        queries = queries.copy(order="C")
        _l2norm_inplace(base)
        _l2norm_inplace(queries)
        index = faiss.IndexFlatIP(base.shape[1])
    else:
        index = faiss.IndexFlatL2(base.shape[1])

    index.add(base)

    # self 模式要去掉自己，因此先多搜 1 个，再过滤自己。
    search_k = int(k_true + 1) if mode == "self" else int(k_true)
    dists, idx_base = index.search(queries, search_k)

    idx_base = idx_base.astype(np.int32, copy=False)
    dists = dists.astype(np.float32, copy=False)

    if mode == "self":
        true_indices = np.empty((queries.shape[0], k_true), dtype=np.int32)
        true_distances = np.empty((queries.shape[0], k_true), dtype=np.float32)
        for i in range(queries.shape[0]):
            mask = idx_base[i] != i
            kept_idx = idx_base[i][mask][:k_true]
            kept_dist = dists[i][mask][:k_true]
            if kept_idx.shape[0] < k_true:
                raise ValueError(
                    f"Not enough non-self neighbors for row {i}: "
                    f"need {k_true}, got {kept_idx.shape[0]}"
                )
            true_indices[i] = kept_idx
            true_distances[i] = kept_dist
    else:
        true_indices = idx_base[:, :k_true]
        true_distances = dists[:, :k_true]

    out_path = str(out_path)
    np.savez_compressed(
        out_path,
        true_indices_base=true_indices,
        true_distances=true_distances,
        source_hdf5=np.array(str(Path(hdf5_path))),
        mode=np.array(mode),
        base_limit=np.int32(base.shape[0]),
        query_limit=np.int32(queries.shape[0]),
        k_true=np.int32(k_true),
        metric=np.array(metric),
    )
    return out_path



def load_true_knn(path: str | Path) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)

    if "true_indices_base" not in z.files:
        raise KeyError(f"Unknown gt npz format. keys={z.files}")

    return {
        "true_indices_base": z["true_indices_base"].astype(np.int32, copy=False),
        "true_distances": z["true_distances"].astype(np.float32, copy=False)
        if "true_distances" in z.files
        else None,
        "source_hdf5": str(z["source_hdf5"]) if "source_hdf5" in z.files else None,
        "mode": str(z["mode"]) if "mode" in z.files else None,
        "base_limit": int(z["base_limit"]) if "base_limit" in z.files else None,
        "query_limit": int(z["query_limit"]) if "query_limit" in z.files else None,
        "k_true": int(z["k_true"]) if "k_true" in z.files else None,
        "metric": str(z["metric"]) if "metric" in z.files else None,
    }



def load_subset_by_gt(gt: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    source_hdf5 = gt.get("source_hdf5")
    mode = gt.get("mode")
    base_limit = gt.get("base_limit")
    query_limit = gt.get("query_limit")
    if not source_hdf5 or not mode:
        raise ValueError("Ground-truth cache does not contain source_hdf5/mode metadata")
    return _load_subset_from_hdf5(
        source_hdf5,
        mode=str(mode),
        base_limit=None if base_limit is None else int(base_limit),
        query_limit=None if query_limit is None else int(query_limit),
    )
