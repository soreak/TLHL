from __future__ import annotations

import numpy as np


def to_float32_matrix(X: np.ndarray) -> np.ndarray:
    """Convert input to a contiguous float32 matrix of shape [N, dim]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("输入数据 X 必须是二维数组 [N, dim]")
    return np.ascontiguousarray(X)


def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance."""
    diff = a - b
    return float(np.dot(diff, diff))
