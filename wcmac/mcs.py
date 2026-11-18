"""MCS table and BLER model (Design §7.1; Protocol §3.3).

    BLER_m(gamma) = 1 / (1 + exp(a_m (gamma - b_m)))
    theta_m       = b_m + ln((1-beta)/beta)/a_m       (BLER_m(theta_m) = beta)
"""

from __future__ import annotations

import numpy as np

from . import config as C


def bler(m: np.ndarray, gamma: np.ndarray, opt=C.DEFAULT_OPTIONS) -> np.ndarray:
    """Vectorised BLER for MCS index array `m` (1-based) and SNR `gamma` (dB)."""
    m = np.asarray(m, dtype=np.int64)
    a = np.asarray(opt.a_slope, dtype=float)[m - 1]
    b = np.asarray(opt.b_mid, dtype=float)[m - 1]
    return 1.0 / (1.0 + np.exp(a * (gamma - b)))


def rate(m: np.ndarray, opt=C.DEFAULT_OPTIONS) -> np.ndarray:
    return np.asarray(opt.rates, dtype=float)[np.asarray(m, dtype=np.int64) - 1]


def select_mcs(gamma_safe: np.ndarray, opt=C.DEFAULT_OPTIONS) -> np.ndarray:
    """m_t = max{k : gamma_safe >= theta_k}, with m = 1 when no theta is met.

    This is the forced-lowest rule of Algorithm Spec §6.3 Step 4.  Implemented as
    a vectorised count so that A, B and C share one selection rule exactly.
    """
    th = np.asarray(opt.theta, dtype=float)
    gs = np.asarray(gamma_safe, dtype=float)
    m = np.ones(gs.shape, dtype=np.int64)
    for k in range(1, len(th)):
        m = np.where(gs >= th[k], k + 1, m)
    return m


def is_aggressive(m: np.ndarray, gamma: np.ndarray, opt=C.DEFAULT_OPTIONS) -> np.ndarray:
    """1{BLER_{m_t}(gamma_t) > beta}.  Under strict monotonicity (T9) this equals
    1{gamma_t < theta_{m_t}}; the definition by BLER is the authoritative one."""
    return (bler(m, gamma, opt) > opt.beta).astype(np.int8)


def is_forced_lowest(gamma_safe: np.ndarray, opt=C.DEFAULT_OPTIONS) -> np.ndarray:
    return (np.asarray(gamma_safe, dtype=float) < opt.theta[0]).astype(np.int8)
