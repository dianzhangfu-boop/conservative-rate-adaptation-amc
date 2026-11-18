"""One-tap linear predictor, fitted once per delay (Algorithm Spec §4).

    gamma_tilde_t = c0^(d) + c1^(d) * gamma_hat_{t-d}

Fitted by OLS on the training traces only.  `fit` is the only place in the
codebase that reads `trace.true`; T6 asserts the trace IDs consumed are all in
the training split, and T1 asserts nothing online reads ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import Trace


@dataclass(frozen=True)
class Predictor:
    c0: float
    c1: float
    d: int
    n_pairs: int

    def predict(self, measured_lagged: np.ndarray) -> np.ndarray:
        return self.c0 + self.c1 * np.asarray(measured_lagged, dtype=float)

    def as_dict(self) -> dict:
        return {"c0": self.c0, "c1": self.c1, "d": self.d, "n_pairs": self.n_pairs}


def fit(traces: list[Trace], d: int, trace_ids: list[str] | None = None,
        ledger: list | None = None) -> Predictor:
    """OLS over (gamma_hat_{s-d}, gamma_s) pairs, s from d+1 to T (0-based: d..T-1)."""
    if d < 1:
        raise ValueError("d must be >= 1 (Algorithm Spec §2.1)")
    xs, ys = [], []
    for tr in traces:
        meas = tr.measured
        xs.append(meas[: tr.true.size - d])
        ys.append(tr.true[d:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if ledger is not None and trace_ids is not None:
        ledger.append({"d": d, "trace_ids": list(trace_ids), "n_pairs": int(x.size)})

    xbar = x.mean()
    ybar = y.mean()
    sxx = np.sum((x - xbar) ** 2)
    sxy = np.sum((x - xbar) * (y - ybar))
    if sxx == 0.0:
        c1 = 0.0
    else:
        c1 = float(sxy / sxx)
    c0 = float(ybar - c1 * xbar)
    return Predictor(c0=c0, c1=c1, d=d, n_pairs=int(x.size))


class PredictorBank:
    """The four frozen predictors, keyed by d; a mismatched d raises (T3)."""

    def __init__(self, by_d: dict[int, Predictor]):
        self.by_d = dict(by_d)

    def get(self, d: int) -> Predictor:
        if d not in self.by_d:
            raise KeyError(
                f"no predictor fitted for d={d}; available {sorted(self.by_d)}"
            )
        p = self.by_d[d]
        if p.d != d:
            raise ValueError(f"predictor d mismatch: requested {d}, stored {p.d}")
        return p

    def as_dict(self) -> dict:
        return {str(d): p.as_dict() for d, p in sorted(self.by_d.items())}
