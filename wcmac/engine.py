"""Online controller, access contract and the four policy runners.

The structural separation required by Algorithm Spec §7 is implemented here:

* ``MeasurementFeed`` is the *only* object the controller touches.  It exposes
  ``measurement_at(instant)`` and raises when the requested instant is later
  than ``current_instant - d``.  It holds no reference to the true SNR.
* ``run_*`` functions receive a ``Trace`` (ground truth) and pass only the feed
  to ``Decide``.  ``trace.true`` is read by the *evaluator* after the decision
  for that slot has been made and logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import mcs
from .predictor import Predictor
from .quantile import empirical_quantile


class LeakageError(RuntimeError):
    """Raised on any attempt to read an unavailable measurement."""


class MeasurementFeed:
    """Causal, read-only view of the measured-SNR sequence.

    Slot indices are 1-based throughout, matching Algorithm Spec §2.1.
    """

    __slots__ = ("_measured", "_d", "_t", "reads")

    def __init__(self, measured: np.ndarray, d: int):
        if d < 1:
            raise ValueError("d must be >= 1")
        self._measured = measured
        self._d = d
        self._t = 0
        self.reads: list[tuple[int, int]] = []

    def set_instant(self, t: int) -> None:
        self._t = t

    def measurement_at(self, instant: int) -> float:
        if instant < 1:
            raise LeakageError(f"instant {instant} precedes slot 1")
        if instant > self._t - self._d:
            raise LeakageError(
                f"attempt to read measurement {instant} at decision instant "
                f"{self._t} with delay {self._d} (available at most {self._t - self._d})"
            )
        self.reads.append((self._t, instant))
        value = float(self._measured[instant - 1])
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite measurement at instant {instant}")
        return value


@dataclass
class ControllerState:
    buffer: np.ndarray
    count: int = 0
    pos: int = 0
    delta: float = 0.0
    pending: dict = field(default_factory=dict)

    @classmethod
    def make(cls, n_w: int, delta_init: float) -> "ControllerState":
        return cls(buffer=np.empty(n_w, dtype=float), delta=float(delta_init))

    def append(self, value: float) -> None:
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite residual {value!r}")
        self.buffer[self.pos] = value
        self.pos = (self.pos + 1) % self.buffer.size
        self.count = min(self.count + 1, self.buffer.size)

    def values(self) -> np.ndarray:
        return self.buffer if self.count == self.buffer.size else self.buffer[: self.count]


def decide_adaptive(t: int, state: ControllerState, predictor: Predictor,
                    opt, feed: MeasurementFeed) -> tuple[float, float, float]:
    """Algorithm Spec §6.3 for the adaptive (Policy C) margin.

    Returns (gamma_tilde, delta, gamma_safe).  Policy A calls this with the
    margin forced to 0 and Policy B with it forced to Delta_fixed; both bypass
    the window but keep the identical prediction path.
    """
    d = predictor.d
    if t <= d:
        gamma_tilde = C.MUHAT0
    else:
        x = feed.measurement_at(t - d)
        gamma_tilde = predictor.c0 + predictor.c1 * x
        state.pending[t] = gamma_tilde

    s = t - d
    if s in state.pending:
        e_new = state.pending.pop(s) - feed.measurement_at(s)
        state.append(e_new)

    if state.count < opt.n_min:
        delta = state.delta
    else:
        q = empirical_quantile(state.values(), opt.alpha)
        delta = min(max(q, 0.0), opt.delta_max)

    return gamma_tilde, delta, gamma_tilde - delta


def _finish(trace, gamma_tilde, delta, opt, keep_slots: bool):
    gamma_safe = gamma_tilde - delta
    m = mcs.select_mcs(gamma_safe, opt)
    p_success = 1.0 - mcs.bler(m, trace.true, opt)
    success = (trace.u <= p_success)
    return _metrics(trace, gamma_tilde, m, delta, gamma_safe, p_success,
                    success, opt, keep_slots)


@dataclass
class Result:
    policy: str
    G: float
    G_real: float
    BLER_obs: float
    F_aggr: float
    mean_mcs: float
    mean_delta: float
    clip_hi_frac: float
    clip_lo_frac: float
    forced_lowest_frac: float
    warmup_slots: int
    # optional per-slot payload, only when keep_slots=True
    slots: dict | None = None

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "slots"}
        return d


def _metrics(trace, gamma_tilde, m, delta, gamma_safe, p_success, success,
             opt, keep_slots) -> Result:
    T = trace.true.size
    r = mcs.rate(m, opt)
    g_model = r * p_success
    aggr = (1.0 - p_success) > opt.beta
    fl = gamma_safe < opt.theta[0]
    hi = delta >= opt.delta_max
    lo = delta <= 0.0
    res = Result(
        policy="",
        G=float(np.mean(g_model)),
        G_real=float(np.mean(r * success)),
        BLER_obs=float(1.0 - np.mean(success)),
        F_aggr=float(np.mean(aggr)),
        mean_mcs=float(np.mean(m)),
        mean_delta=float(np.mean(delta)),
        clip_hi_frac=float(np.mean(hi)),
        clip_lo_frac=float(np.mean(lo)),
        forced_lowest_frac=float(np.mean(fl)),
        warmup_slots=0,          # set by each runner (cold start vs full warm-up)
    )
    if keep_slots:
        res.slots = {
            "gamma_tilde": np.asarray(gamma_tilde, dtype=float),
            "m": m.astype(np.int16),
            "delta": delta.astype(np.float64),
            "gamma_safe": gamma_safe.astype(np.float64),
            "aggr": aggr.astype(np.int8),
            "success": success.astype(np.int8),
            "G_model": g_model,
        }
    return res


def result_from_path(trace, gamma_tilde, delta, opt=C.DEFAULT_OPTIONS,
                     policy: str = "C", keep_slots: bool = False) -> Result:
    """Rebuild a Result for a modified margin path without re-running the loop.

    Valid only where the margin trajectory is independent of the change -- in
    particular, replacing Delta_init affects only the warm-up slots whose
    decisions are already fully described by (gamma_tilde, delta).
    """
    res = _finish(trace, np.asarray(gamma_tilde, float), np.asarray(delta, float),
                  opt, keep_slots)
    res.policy = policy
    return res


# --------------------------------------------------------------------------
# Policy runners
# --------------------------------------------------------------------------


def _predictions(trace, predictor: Predictor) -> np.ndarray:
    """gamma_tilde for every slot: cold-start constant for t <= d, then one tap."""
    d = predictor.d
    T = trace.true.size
    gt = np.empty(T, dtype=float)
    gt[:d] = C.MUHAT0
    if T > d:
        gt[d:] = predictor.c0 + predictor.c1 * trace.measured[: T - d]
    return gt


def run_constant_margin(trace, predictor: Predictor, delta_fixed: float,
                        opt=C.DEFAULT_OPTIONS, policy: str = "B",
                        keep_slots: bool = False) -> Result:
    """Policy A (Delta=0) and Policy B (Delta=Delta_fixed).

    One code path: the margin is a constant for every slot, warm-up included,
    which is exactly the condition under which A == B(0) (Protocol §6.2, T5).
    """
    gt = _predictions(trace, predictor)
    # The margin is a constant for every slot, warm-up included: no separate
    # warm-up branch exists, which is condition (ii) of A == B(0).
    delta = np.full(trace.true.size, float(delta_fixed))
    res = _finish(trace, gt, delta, opt, keep_slots)
    res.policy = policy
    res.warmup_slots = predictor.d
    return res


def run_adaptive(trace, predictor: Predictor, delta_init: float,
                 opt=C.DEFAULT_OPTIONS, policy: str = "C",
                 keep_slots: bool = False, record: bool = False,
                 feed_cls=MeasurementFeed) -> Result:
    T = trace.true.size
    d = predictor.d
    meas = trace.measured
    state = ControllerState.make(opt.n_w, delta_init)
    feed = feed_cls(meas, d)

    gamma_tilde = np.empty(T)
    delta = np.empty(T)
    pending: dict[int, float] = {}
    residual_log = np.full(T, np.nan) if record else None
    events: dict = {}

    # The loop below is the literal transcription of Algorithm Spec §6.3; the
    # feed guard makes any future read raise rather than return a value.
    for t in range(1, T + 1):
        feed.set_instant(t)
        if t <= d:
            gtilde = C.MUHAT0
        else:
            x = feed.measurement_at(t - d)
            gtilde = predictor.c0 + predictor.c1 * x
            pending[t] = gtilde

        s = t - d
        if s in pending:
            e_new = pending.pop(s) - feed.measurement_at(s)
            state.append(e_new)
            if record:
                residual_log[s - 1] = e_new
                events.setdefault("first_residual_instant", t)

        if state.count < opt.n_min:
            dlt = state.delta
        else:
            if record and "handover_instant" not in events:
                events["handover_instant"] = t
            dlt = empirical_quantile(state.values(), opt.alpha)
            if dlt > opt.delta_max:
                dlt = opt.delta_max
            elif dlt < 0.0:
                dlt = 0.0
            if record and state.count == opt.n_w and "window_full_instant" not in events:
                events["window_full_instant"] = t

        gamma_tilde[t - 1] = gtilde
        delta[t - 1] = dlt

    res = _finish(trace, gamma_tilde, delta, opt, keep_slots)
    res.policy = policy
    res.warmup_slots = 2 * d + opt.n_min
    if record:
        events["n_reads"] = len(feed.reads)
        events["max_read"] = max(feed.reads) if feed.reads else 0
        events["reads"] = list(feed.reads)
        res.slots = res.slots or {}
        res.slots.update({"residual": residual_log,
                          "gamma_tilde": gamma_tilde,
                          "events": events})
    return res


def olla_step(delta: float, nack: bool, opt=C.DEFAULT_OPTIONS,
              k_down: float = C.DELTA_DOWN, k_up: float = C.DELTA_UP) -> float:
    """One OLLA margin update (Algorithm Spec §6.6).  NACK raises the margin."""
    if nack:
        return min(delta + k_down, opt.delta_max)
    return max(delta - k_up, 0.0)


def run_olla(trace, predictor: Predictor, delta_init: float,
             opt=C.DEFAULT_OPTIONS, k_down: float = C.DELTA_DOWN,
             k_up: float = C.DELTA_UP, policy: str = "D",
             keep_slots: bool = False,
             forced_nack: np.ndarray | None = None) -> Result:
    """Policy D: outcome-driven margin update (Algorithm Spec §6.6)."""
    T = trace.true.size
    d = predictor.d
    meas = trace.measured
    u = trace.u
    theta = opt.theta
    feed = MeasurementFeed(meas, d)

    gamma_tilde = np.empty(T)
    delta = np.empty(T)
    m_arr = np.empty(T, dtype=np.int64)
    success = np.empty(T, dtype=bool)
    nack = np.zeros(T, dtype=bool)

    dlt = float(delta_init)
    for t in range(1, T + 1):
        feed.set_instant(t)

        # ---- update from the outcome of slot t-d (available now, before the
        #      decision for slot t; Algorithm Spec §6.6 "Timing") ----
        j = t - d
        if 1 <= j <= T:
            src = forced_nack if forced_nack is not None else nack
            if src[j - 1]:
                dlt = min(dlt + k_down, opt.delta_max)
            else:
                dlt = max(dlt - k_up, 0.0)

        if t <= d:
            gtilde = C.MUHAT0
        else:
            gtilde = predictor.c0 + predictor.c1 * feed.measurement_at(t - d)

        gamma_safe = gtilde - dlt
        m = 1
        for k in range(1, len(theta)):
            if gamma_safe >= theta[k]:
                m = k + 1
        # p_success = 1 - BLER = 1 / (1 + exp(-a (gamma - b)))  -- note the sign
        p_success = 1.0 / (1.0 + np.exp(-opt.a_slope[m - 1] * (trace.true[t - 1] - opt.b_mid[m - 1])))
        ok = bool(u[t - 1] <= p_success)
        nack[t - 1] = not ok

        gamma_tilde[t - 1] = gtilde
        delta[t - 1] = dlt
        m_arr[t - 1] = m
        success[t - 1] = ok

    p_success = 1.0 - mcs.bler(m_arr, trace.true, opt)
    res = _metrics(trace, gamma_tilde, m_arr, delta, gamma_tilde - delta,
                   p_success, success, opt, keep_slots)
    res.policy = policy
    res.warmup_slots = d
    return res
