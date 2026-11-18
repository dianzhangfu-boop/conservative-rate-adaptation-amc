"""E5 segment and recovery-time metrics (Design §3 H5; Protocol §7.2)."""

from __future__ import annotations

import numpy as np

from . import config as C


def segment_means(aggr: np.ndarray) -> dict:
    """F_aggr over the pre-step, transient and post-step windows."""
    return {
        "F_pre": float(np.mean(aggr[C.PRE_STEP[0]: C.PRE_STEP[1]])),
        "F_trans": float(np.mean(aggr[C.TRANSIENT[0]: C.TRANSIENT[1]])),
        "F_post": float(np.mean(aggr[C.POST_STEP[0]: C.POST_STEP[1]])),
    }


def recovery_time(aggr: np.ndarray) -> tuple[float, bool]:
    """tau = min{u in [t_step+99, T-99] : |F_trail(v) - F_post| <= 0.02
    for every v in [u, u+99]}, else tau = T with censored = True.

    F_trail(v) is the indicator mean over the trailing 100-slot window [v-99, v].
    """
    aggr = np.asarray(aggr, dtype=float)
    n = aggr.size
    w = C.RECOVERY_WINDOW
    post = float(np.mean(aggr[C.POST_STEP[0]: C.POST_STEP[1]]))

    trail = np.convolve(aggr, np.ones(w) / w, mode="valid")   # trail[i] -> v = i + w
    ok = np.abs(trail - post) <= C.RECOVERY_BAND

    run = np.convolve(ok.astype(np.int64), np.ones(C.RECOVERY_SUSTAIN, dtype=np.int64),
                      mode="valid") == C.RECOVERY_SUSTAIN
    js = np.flatnonzero(run)
    # u = j + w, u >= t_step + 99 -> j >= t_step - 1; u <= T - 99 -> j <= T - 199
    lo_i = C.T_STEP - 1
    hi_i = C.T_SLOTS - 199
    js = js[(js >= lo_i) & (js <= hi_i)]
    if js.size == 0:
        return float(n), True
    return float(js[0] + w), False


def prediction_rmse(pred: np.ndarray, true: np.ndarray, d: int) -> float:
    """RMSE of the predicted vs true SNR over slots t > d (diagnostic only)."""
    return float(np.sqrt(np.mean((pred[d:] - true[d:]) ** 2)))
