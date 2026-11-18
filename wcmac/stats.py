"""Paired statistics and the pre-registered three-state decision rules.

Delta convention (Design §3 intro, authoritative):  dX = X_C - X_baseline.
Every confidence interval is pointwise 95 %; no family-wise control is claimed
beyond the Holm families of Protocol §8.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

SUPPORTED = "Supported"
CONTRADICTED = "Contradicted"
INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True)
class PairedCI:
    n: int
    mean: float
    sd: float
    se: float
    half: float
    lo: float
    hi: float
    t: float
    p: float
    dz: float
    df: int

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("n", "mean", "sd", "se", "half", "lo", "hi", "t", "p", "dz", "df")}


def paired_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> PairedCI:
    """CI of the mean of a - b across traces (paired, within-trace)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    diff = a - b
    n = diff.size
    if n < 2:
        raise ValueError("need at least two pairs")
    df = n - 1
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(sps.t.ppf(0.5 + level / 2.0, df))
    half = tcrit * se
    tstat = mean / se if se > 0 else (0.0 if mean == 0 else float(np.sign(mean) * np.inf))
    p = float(2.0 * sps.t.sf(abs(tstat), df)) if np.isfinite(tstat) else 0.0
    dz = mean / sd if sd > 0 else 0.0
    return PairedCI(n=n, mean=mean, sd=sd, se=se, half=half, lo=mean - half,
                    hi=mean + half, t=float(tstat), p=p, dz=float(dz), df=df)


def welch_ci(a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray,
             df_fixed: int = 9, level: float = 0.95) -> PairedCI:
    """CI of D(2) - D(1) where D = a - b within each independent cell.

    The design fixes the conservative df = 9 rather than Welch-Satterthwaite
    (Design §3, H3).
    """
    d1 = np.asarray(a1) - np.asarray(b1)
    d2 = np.asarray(a2) - np.asarray(b2)
    n1, n2 = d1.size, d2.size
    mean = float(d2.mean() - d1.mean())
    var = d1.var(ddof=1) / n1 + d2.var(ddof=1) / n2
    se = float(np.sqrt(var))
    tcrit = float(sps.t.ppf(0.5 + level / 2.0, df_fixed))
    half = tcrit * se
    tstat = mean / se if se > 0 else (0.0 if mean == 0 else float(np.sign(mean) * np.inf))
    p = float(2.0 * sps.t.sf(abs(tstat), df_fixed)) if np.isfinite(tstat) else 0.0
    sd = float(np.sqrt(d1.var(ddof=1) + d2.var(ddof=1)))
    dz = mean / sd if sd > 0 else 0.0
    return PairedCI(n=n1 + n2, mean=mean, sd=sd, se=se, half=half, lo=mean - half,
                    hi=mean + half, t=float(tstat), p=p, dz=float(dz), df=df_fixed)


# --------------------------------------------------------------------------
# Three-state rules (Design §3).  Each returns one of the three strings.
# --------------------------------------------------------------------------


def class_lower(lo: float, hi: float) -> str:
    """Expected direction: quantity < 0."""
    if hi < 0.0:
        return SUPPORTED
    if lo >= 0.0:
        return CONTRADICTED
    return INCONCLUSIVE


def class_upper(lo: float, hi: float) -> str:
    """Expected direction: quantity > 0."""
    if lo > 0.0:
        return SUPPORTED
    if hi <= 0.0:
        return CONTRADICTED
    return INCONCLUSIVE


def class_at_most(lo: float, hi: float, bound: float) -> str:
    """Tolerance leg: quantity <= bound."""
    if hi <= bound:
        return SUPPORTED
    if lo > bound:
        return CONTRADICTED
    return INCONCLUSIVE


def class_inside(lo: float, hi: float, bound: float) -> str:
    """Two-sided tolerance leg: |quantity| <= bound."""
    if lo >= -bound and hi <= bound:
        return SUPPORTED
    if lo > bound or hi < -bound:
        return CONTRADICTED
    return INCONCLUSIVE


def joint(states: list[str]) -> str:
    if all(s == SUPPORTED for s in states):
        return SUPPORTED
    if any(s == CONTRADICTED for s in states):
        return CONTRADICTED
    return INCONCLUSIVE


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down. Returns {key: {'p': p, 'reject': bool, 'thr': float}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    still = True
    for rank, (key, p) in enumerate(items):
        thr = 0.05 / (m - rank)
        reject = bool(still and p <= thr)
        if not reject:
            still = False
        out[key] = {"p": float(p), "reject": reject, "thr": float(thr)}
    return out
