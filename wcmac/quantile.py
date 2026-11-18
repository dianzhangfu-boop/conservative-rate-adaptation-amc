"""Empirical quantile, type-7 convention (Algorithm Spec §5.3, T8).

    h = (m - 1) * alpha + 1
    j = floor(h), g = h - j
    Q = x_(j) + g * (x_(j+1) - x_(j)),   x_(m+1) := x_(m)

0-based equivalent used below:
    idx = (m - 1) * alpha; k = floor(idx); g = idx - k
    Q = v[k] + g * (v[k+1] - v[k])   (k clamped to m-1)
"""

from __future__ import annotations

import numpy as np


def empirical_quantile(values: np.ndarray, alpha: float) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    m = v.size
    if m < 1:
        raise ValueError("empty window")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha out of range: {alpha}")
    idx = (m - 1) * alpha
    k = int(np.floor(idx))
    g = idx - k
    if k >= m - 1:
        return float(v[m - 1])
    return float(v[k] + g * (v[k + 1] - v[k]))
