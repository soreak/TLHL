# from __future__ import annotations
#
# import numpy as np
#
#
# def to_float32_matrix(X: np.ndarray) -> np.ndarray:
#     """Convert input to a contiguous float32 matrix of shape [N, dim]."""
#     X = np.asarray(X, dtype=np.float32)
#     if X.ndim != 2:
#         raise ValueError("输入数据 X 必须是二维数组 [N, dim]")
#     return np.ascontiguousarray(X)
#
#
# def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
#     """Squared L2 distance."""
#     diff = a - b
#     return float(np.dot(diff, diff))
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(func):
            return func
        return deco


def to_float32_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("输入数据 X 必须是二维数组 [N, dim]")
    return np.ascontiguousarray(X)


@njit(cache=True, fastmath=True)
def _l2_sq_numba(a: np.ndarray, b: np.ndarray) -> float:
    s = 0.0
    for i in range(a.shape[0]):
        d = float(a[i]) - float(b[i])
        s += d * d
    return s


@njit(cache=True, fastmath=True)
def _row_l2_sq_numba(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    d = matrix.shape[1]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = 0.0
        for j in range(d):
            diff = float(query[j]) - float(matrix[i, j])
            s += diff * diff
        out[i] = s
    return out


def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    """
    单对向量 squared L2。
    - 有 numba 时走 JIT kernel
    - 无 numba 时走 numpy fallback
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if HAS_NUMBA:
        return float(_l2_sq_numba(a, b))

    diff = a - b
    return float(np.dot(diff, diff))


def row_l2_sq(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    计算 query 到 matrix 每一行的 squared L2 distance。
    返回 shape = [N] 的 float32 数组。
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    M = np.asarray(matrix, dtype=np.float32)

    if M.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    if M.shape[1] != q.shape[0]:
        raise ValueError("query 维度与 matrix 不一致")

    M = np.ascontiguousarray(M)
    q = np.ascontiguousarray(q)

    if HAS_NUMBA:
        return _row_l2_sq_numba(q, M)

    diff = M - q[None, :]
    return np.sum(diff * diff, axis=1, dtype=np.float32).astype(np.float32, copy=False)

def l2_sq_fast(a: np.ndarray, b: np.ndarray) -> float:
    """
    更轻的单对向量 squared L2。
    假设 a / b 已经是 shape=[dim] 的 float32 向量。
    这个函数不给通用调用方做 np.asarray 转换，专门给热路径用。
    """
    if HAS_NUMBA:
        return float(_l2_sq_numba(a, b))

    diff = a - b
    return float(np.dot(diff, diff))


def prepare_query_vector(query: np.ndarray) -> np.ndarray:
    """
    给热路径使用的轻量预处理：
    - 转成 float32
    - reshape(-1)
    - contiguous
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(q)