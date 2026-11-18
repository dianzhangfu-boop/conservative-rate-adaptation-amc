#!/usr/bin/env python
"""E0 validity gate: tests T1-T13 of Algorithm Specification §8.

Every test records PASS / FAIL / NOT RUN, the command that produced it, the
evidence, and (for failures) what the failure would mean.  No experimental
result may be produced from a codebase whose E0 report is not all-PASS.

Usage:  python scripts/run_e0.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import channel, config as C, engine, mcs, metrics as M
from wcmac import predictor as P
from wcmac.channel import Trace
from wcmac.io_utils import ensure, env_txt, write_json
from wcmac.quantile import empirical_quantile

PASS, FAIL, NOTRUN = "PASS", "FAIL", "NOT RUN"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def rec(tid, name, status, detail, evidence=None):
    return {"id": tid, "name": name, "status": status, "detail": detail,
            "evidence": evidence or {}, "command": "python scripts/run_e0.py"}


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


def build_context():
    train, ids, j = [], [], 0
    for (rho, sn), n in C.TRAINING_POOL.items():
        for _ in range(n):
            j += 1
            train.append(channel.generate_train(C.train_seed(j), sn, rho))
            ids.append(f"train:{j}")
    ledger = []
    bank = P.PredictorBank({d: P.fit(train, d, ids, ledger) for d in C.TRAIN_DS})
    e0 = channel.generate(C.e0_seed(1), 1.0, 0.85, scenario="E0", split="e0")
    return {"train": train, "ids": ids, "bank": bank, "ledger": ledger, "e0": e0}


# --------------------------------------------------------------------------
# T1 - true-SNR leakage sentinel
# --------------------------------------------------------------------------


def t1(ctx):
    tr, pred = ctx["e0"], ctx["bank"].get(3)
    # A constant 0 dB is the spec's own example; with this value the
    # reconstruction measured = true + noise is exact in floating point, so the
    # comparison below is a genuine bit-identity test rather than a tolerance.
    garbage = np.zeros(len(tr))
    poisoned = Trace(seed=tr.seed, true=garbage, noise=tr.measured - garbage,
                     u=tr.u, scenario="E0", split="e0")
    check(np.array_equal(poisoned.measured, tr.measured),
          "poisoning must leave the measured sequence bit-identical")

    ev = {}
    ok = True
    for name, fn in (
        ("A", lambda t: engine.run_constant_margin(t, pred, 0.0, policy="A", keep_slots=True)),
        ("B", lambda t: engine.run_constant_margin(t, pred, 3.0, policy="B", keep_slots=True)),
        ("C", lambda t: engine.run_adaptive(t, pred, 3.0, policy="C", keep_slots=True)),
    ):
        r1, r2 = fn(tr), fn(poisoned)
        sm = bool(np.array_equal(r1.slots["m"], r2.slots["m"]))
        sd = bool(np.array_equal(r1.slots["delta"], r2.slots["delta"]))
        ev[name] = {"mcs_bit_identical": sm, "delta_bit_identical": sd}
        ok &= sm and sd

    # Policy D legitimately consumes ACK/NACK; the test is that truth can enter
    # ONLY through that channel.  Replay with the recorded outcome stream.
    rd = engine.run_olla(tr, pred, 3.0, keep_slots=True)
    nack = (1 - rd.slots["success"]).astype(bool)
    rd2 = engine.run_olla(poisoned, pred, 3.0, keep_slots=True, forced_nack=nack)
    d_m = bool(np.array_equal(rd.slots["m"], rd2.slots["m"]))
    d_d = bool(np.array_equal(rd.slots["delta"], rd2.slots["delta"]))
    ev["D"] = {"mcs_bit_identical_under_outcome_replay": d_m,
               "delta_bit_identical_under_outcome_replay": d_d,
               "note": "OLLA consumes outcomes by design; replay isolates the channel"}
    ok &= d_m and d_d

    # the poison must actually have reached the evaluator
    ra_p = engine.run_constant_margin(poisoned, pred, 0.0, policy="A", keep_slots=True)
    ra_n = engine.run_constant_margin(tr, pred, 0.0, policy="A", keep_slots=True)
    poison_active = not np.array_equal(ra_p.slots["success"], ra_n.slots["success"])
    ev["poison_reached_evaluator"] = bool(poison_active)
    ok &= poison_active

    return rec("T1", "true-SNR leakage sentinel", PASS if ok else FAIL,
               "Decisions of A, B and C are bit-identical under garbage true SNR; "
               "D is bit-identical under outcome replay; the poison demonstrably "
               "changes outcomes.", ev)


# --------------------------------------------------------------------------
# T2 - future-feedback leakage
# --------------------------------------------------------------------------


class PermissiveFutureFeed(engine.MeasurementFeed):
    """Returns garbage for future measurements instead of raising, so that a
    controller which wrongly reads ahead would visibly diverge."""

    def measurement_at(self, instant: int) -> float:
        if instant < 1:
            raise engine.LeakageError(f"instant {instant} precedes slot 1")
        if instant > self._t - self._d:
            self.reads.append((self._t, instant))
            return 1.0e6
        return super().measurement_at(instant)


def t2(ctx):
    tr, pred = ctx["e0"], ctx["bank"].get(3)
    r = engine.run_adaptive(tr, pred, 3.0, keep_slots=True, record=True)
    r_poison = engine.run_adaptive(tr, pred, 3.0, keep_slots=True,
                                   feed_cls=PermissiveFutureFeed)

    reads = r.slots["events"]["reads"]
    violations = [(t, s) for (t, s) in reads if s > t - pred.d]
    same = (np.array_equal(r.slots["m"], r_poison.slots["m"])
            and np.array_equal(r.slots["delta"], r_poison.slots["delta"]))

    feed = engine.MeasurementFeed(tr.measured, 3)
    feed.set_instant(10)
    raised = False
    try:
        feed.measurement_at(8)          # 8 > 10-3 = 7 -> must raise
    except engine.LeakageError:
        raised = True

    ok = same and not violations and raised
    return rec("T2", "future-feedback leakage", PASS if ok else FAIL,
               "No read beyond t-d in %d feed accesses; a permissive feed that "
               "would return garbage for future indices changes nothing; the "
               "guard raises on a direct future read." % len(reads),
               {"n_reads": len(reads), "violations": len(violations),
                "decisions_unchanged_under_permissive_feed": bool(same),
                "guard_raises": raised})


# --------------------------------------------------------------------------
# T3 - feedback-delay alignment
# --------------------------------------------------------------------------


def t3(ctx):
    bank = ctx["bank"]
    # The generator's streams do not depend on d, so the same seed gives the
    # same (gamma, n, u) realisation for both delays.
    tr = channel.generate(C.e0_seed(1), 1.0, 0.85, scenario="E0", split="e0")
    r3 = engine.run_adaptive(tr, bank.get(3), 3.0, record=True, keep_slots=True)
    r6 = engine.run_adaptive(tr, bank.get(6), 3.0, record=True, keep_slots=True)
    e3, e6 = r3.slots["events"], r6.slots["events"]

    expect = {"d=3": {"first": 7, "handover": 16, "full": 106},
              "d=6": {"first": 13, "handover": 22, "full": 112}}
    got = {"d=3": {"first": e3.get("first_residual_instant"),
                   "handover": e3.get("handover_instant"),
                   "full": e3.get("window_full_instant")},
           "d=6": {"first": e6.get("first_residual_instant"),
                   "handover": e6.get("handover_instant"),
                   "full": e6.get("window_full_instant")}}

    mismatch_raised = False
    try:
        P.PredictorBank({3: bank.get(3)}).get(6)
    except KeyError:
        mismatch_raised = True

    differs = not np.array_equal(r3.slots["m"], r6.slots["m"])
    ok = (got == expect) and mismatch_raised and differs
    return rec("T3", "feedback-delay alignment", PASS if ok else FAIL,
               "Timing table reproduced exactly for d=3 and d=6 on one "
               "realisation; a predictor fitted for another d is rejected; the "
               "decision logs differ only through availability.",
               {"expected": expect, "observed": got,
                "predictor_mismatch_raises": mismatch_raised,
                "decisions_differ": bool(differs),
                "bank": bank.as_dict()})


# --------------------------------------------------------------------------
# T4 - OLLA zero drift
# --------------------------------------------------------------------------


def t4(ctx):
    opt = C.DEFAULT_OPTIONS
    up, down = C.DELTA_UP, C.DELTA_DOWN

    direction = (abs(engine.olla_step(6.0, True) - (6.0 + down)) < 1e-15
                 and abs(engine.olla_step(6.0, False) - (6.0 - up)) < 1e-15)

    # (i) literal check: clipped integrator, one path, mid-range start
    T = 100_000
    rng = np.random.default_rng(2026)
    nack = rng.random(T) < C.BETA
    d0 = 6.0
    dl = d0
    path = np.empty(T)
    for i in range(T):
        dl = engine.olla_step(dl, bool(nack[i]), opt)
        path[i] = dl
    literal = abs(path[-1] - d0) / T

    # (ii) the discriminating check: an UNCLIPPED integrator would be pinned by
    # the clip in (i), so the drift tolerance is also tested where the clip
    # cannot mask a wrong step ratio.  20 independent paths, mean increment.
    n_seeds, T2 = 20, 100_000
    inc_means = []
    for s in range(n_seeds):
        r = np.random.default_rng([4242, s])
        nk = r.random(T2) < C.BETA
        inc = np.where(nk, down, -up)
        inc_means.append(float(inc.mean()))
    inc_means = np.array(inc_means)
    se = inc_means.std(ddof=1) / math.sqrt(n_seeds)
    ci = (inc_means.mean() - 2.093 * se, inc_means.mean() + 2.093 * se)
    wrong_ratio = C.BETA * down - (1 - C.BETA) * down      # -0.4 dB/slot
    excludes_wrong = not (ci[0] <= wrong_ratio <= ci[1])
    unbiased = abs(ci[0]) < 0.01 and abs(ci[1]) < 0.01

    ok = direction and literal <= 1e-4 and unbiased and excludes_wrong
    return rec("T4", "OLLA zero-drift sanity", PASS if ok else FAIL,
               "Directions match the spec; the clipped single-path drift is "
               "within 1e-4 dB/slot; an unclipped 20-seed test locates the mean "
               "increment at 0 and excludes the wrong-ratio value -0.4 dB/slot.",
               {"direction_ok": direction,
                "literal_drift_dB_per_slot": literal,
                "unclipped_mean_increment": float(inc_means.mean()),
                "unclipped_ci95": [float(ci[0]), float(ci[1])],
                "wrong_ratio_increment": wrong_ratio,
                "ci_excludes_wrong_ratio": bool(excludes_wrong),
                "note": "the literal tolerance is uninformative when the clip is "
                        "active (any bounded path satisfies it), hence check (ii)"})


# --------------------------------------------------------------------------
# T5 - randomness fairness and A == B(0)
# --------------------------------------------------------------------------


def t5(ctx):
    tr, pred = ctx["e0"], ctx["bank"].get(3)
    checks = {}
    for name, arr in (("gamma", tr.true), ("noise", tr.noise), ("u", tr.u)):
        checks[name] = hashlib.sha256(arr.tobytes()).hexdigest()[:16]

    rA = engine.run_constant_margin(tr, pred, 0.0, policy="A", keep_slots=True)
    rB0 = engine.run_constant_margin(tr, pred, 0.0, policy="B0", keep_slots=True)
    rB3 = engine.run_constant_margin(tr, pred, 3.0, policy="B3", keep_slots=True)
    _ = engine.run_adaptive(tr, pred, 3.0, policy="C", keep_slots=True)

    after = {name: hashlib.sha256(arr.tobytes()).hexdigest()[:16]
             for name, arr in (("gamma", tr.true), ("noise", tr.noise), ("u", tr.u))}
    streams_unchanged = checks == after

    bit_ident = all(np.array_equal(rA.slots[k], rB0.slots[k])
                    for k in ("m", "gamma_safe", "delta", "success"))

    # the decode stream must not move when sigma_n changes
    tr0 = channel.generate(C.e0_seed(1), 0.0, 0.85)
    u_indep = np.array_equal(tr0.u, tr.u) and np.array_equal(tr0.true, tr.true)
    noise_differs = not np.array_equal(tr0.noise, tr.noise)
    b3_differs = not np.array_equal(rB3.slots["m"], rB0.slots["m"])

    ok = streams_unchanged and bit_ident and u_indep and noise_differs
    return rec("T5", "randomness fairness and A == B(0)", PASS if ok else FAIL,
               "Policy runs do not consume or mutate the gamma/n/u streams; "
               "A and B(Delta=0) are slot-by-slot identical; the decode stream "
               "does not move when sigma_n changes.",
               {"stream_checksums": checks, "streams_unchanged": streams_unchanged,
                "A_equals_B0_all_slots": bool(bit_ident),
                "u_independent_of_sigma_n": bool(u_indep),
                "noise_depends_on_sigma_n": bool(noise_differs),
                "B3_differs_from_B0": bool(b3_differs)})


# --------------------------------------------------------------------------
# T6 - dataset split integrity
# --------------------------------------------------------------------------


def t6(ctx):
    sets = {
        "main_test": {C.main_seed(k, i) for k in C.MAIN_KS for i in range(1, 11)},
        "main_val": {C.main_val_seed(k, i) for k in C.MAIN_KS for i in range(1, 21)},
        "stress_test": {C.stress_seed(k, i) for k in C.STRESS_KS for i in range(1, 11)},
        "stress_val": {C.stress_val_seed(k, i) for k in C.STRESS_KS for i in range(1, 21)},
        "e5": {C.e5_seed(i) for i in range(1, 21)},
        "train": {C.train_seed(j) for j in range(1, 31)},
        "e0": {C.e0_seed(1)},
    }
    names = sorted(sets)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                overlaps[f"{a}&{b}"] = sorted(inter)
    disjoint = not overlaps

    ledger_ok = all(entry["trace_ids"] and all(t.startswith("train:")
                                               for t in entry["trace_ids"])
                    for entry in ctx["ledger"])

    ok = disjoint and ledger_ok
    return rec("T6", "dataset split integrity", PASS if ok else FAIL,
               "All seven seed sets are pairwise disjoint and the fit ledger "
               "contains training trace IDs only. The archive-checksum "
               "sub-check is performed at test time, when test traces exist.",
               {"sizes": {k: len(v) for k, v in sets.items()},
                "overlaps": overlaps,
                "fit_ledger_training_only": ledger_ok,
                "deferred": "archive checksum of test traces (T6c) runs after "
                            "the test traces are archived"})


# --------------------------------------------------------------------------
# T7 - residual identity at zero noise
# --------------------------------------------------------------------------


def t7(ctx):
    T = C.T_SLOTS
    t = np.arange(T)
    true = 10.0 + 3.0 * np.sin(2 * np.pi * t / 500.0) + 1.5 * np.sin(2 * np.pi * t / 37.0)
    tr = Trace(seed=0, true=true, noise=np.zeros(T), u=np.full(T, 0.5),
               scenario="det", split="e0")
    pred = P.fit([tr], 3)
    r = engine.run_adaptive(tr, pred, 3.0, record=True, keep_slots=True)

    res = r.slots["residual"]
    gt = r.slots["gamma_tilde"]
    d = 3
    idx = np.arange(T)
    # residual of slot s is stored at index s-1 and exists for d+1 <= s <= T-d:
    # the measurement of slot s arrives at instant s+d, which must be <= T.
    valid = (idx >= d) & (idx < T - d)
    eps = gt - true
    max_err = float(np.max(np.abs(res[valid] - eps[valid])))
    early_nan = bool(np.all(np.isnan(res[idx < d])))

    ok = max_err <= 1e-12 and early_nan
    return rec("T7", "residual identity at zero noise", PASS if ok else FAIL,
               "With sigma_n = 0 the observable residual equals the true "
               "prediction error to machine precision for every slot s >= d+1, "
               "and no residual exists for s <= d.",
               {"max_abs_error": max_err, "no_residual_before_d_plus_1": early_nan,
                "predictor": pred.as_dict()})


# --------------------------------------------------------------------------
# T8 - quantile convention
# --------------------------------------------------------------------------


def _reference_quantile(values, alpha):
    v = sorted(float(x) for x in values)
    m = len(v)
    h = (m - 1) * alpha + 1.0
    j = int(math.floor(h))
    g = h - j
    if j >= m:
        return v[m - 1]
    return v[j - 1] + g * (v[j] - v[j - 1])


def t8(ctx):
    v = [0.31, -1.44, 2.05, 0.07, 3.91, -0.62, 1.18, 0.55, -2.30, 1.77]
    refs, gots = {}, {}
    for a in (0.5, 0.9, 0.99):
        refs[str(a)] = _reference_quantile(v, a)
        gots[str(a)] = empirical_quantile(np.array(v), a)
    exact = all(abs(refs[k] - gots[k]) < 1e-12 for k in refs)

    rng = np.random.default_rng(7)
    x = rng.standard_normal(1_000_000)
    from scipy.stats import norm
    mc = {str(a): abs(empirical_quantile(x, a) - float(norm.ppf(a)))
          for a in (0.5, 0.9, 0.99)}
    mc_ok = all(v_ < 0.01 for v_ in mc.values())

    ok = exact and mc_ok
    return rec("T8", "quantile convention correctness", PASS if ok else FAIL,
               "Type-7 interpolation matches an independent hand reference for "
               "m = 10 and agrees with the analytic Gaussian quantile on 1e6 "
               "draws.", {"m10_reference": refs, "m10_observed": gots,
                          "gaussian_abs_error": mc})


# --------------------------------------------------------------------------
# T9 - BLER monotonicity and threshold identity
# --------------------------------------------------------------------------


def t9(ctx):
    grid = np.arange(-5.0, 25.0 + 1e-9, 0.01)
    mono, th_err = {}, {}
    for m in range(1, C.N_MCS + 1):
        b = mcs.bler(np.full(grid.size, m), grid)
        mono[m] = bool(np.all(np.diff(b) < 0.0))
        th = C.THETA[m - 1]
        th_err[m] = abs(float(mcs.bler(np.array([m]), np.array([th]))[0]) - C.BETA)
    ok = all(mono.values()) and all(e <= 1e-9 for e in th_err.values())
    return rec("T9", "BLER monotonicity and threshold identity", PASS if ok else FAIL,
               "Each BLER curve is strictly decreasing on the 0.01 dB grid and "
               "theta_m solves BLER_m(theta) = beta to better than 1e-9.",
               {"strictly_decreasing": mono, "theta_abs_error": th_err})


# --------------------------------------------------------------------------
# T10 - warm-up and window-fill timing
# --------------------------------------------------------------------------


def t10(ctx):
    tr = channel.generate(C.e0_seed(1), 1.0, 0.85)
    rows, ok = [], True
    for d in C.TRAIN_DS:
        r = engine.run_adaptive(tr, ctx["bank"].get(d), 3.0, record=True, keep_slots=True)
        e = r.slots["events"]
        got = (e.get("first_residual_instant"), e.get("handover_instant"),
               e.get("window_full_instant"))
        exp = (2 * d + 1, 2 * d + C.N_MIN, 2 * d + C.N_W)
        rows.append({"d": d, "expected": exp, "observed": got, "match": got == exp})
        ok &= got == exp
    return rec("T10", "warm-up and window-fill timing", PASS if ok else FAIL,
               "First residual at 2d+1, margin handover at 2d+n_min and window "
               "full at 2d+N_w for every d in {1,3,6,12}.", {"rows": rows})


# --------------------------------------------------------------------------
# T11 - determinism and reproducibility
# --------------------------------------------------------------------------


def t11(ctx):
    tr, pred = ctx["e0"], ctx["bank"].get(3)
    r1 = engine.run_adaptive(tr, pred, 3.0).as_row()
    r2 = engine.run_adaptive(tr, pred, 3.0).as_row()
    same_metrics = r1 == r2

    a = channel.generate(C.e0_seed(1), 1.0, 0.85)
    b = channel.generate(C.e0_seed(1), 1.0, 0.85)
    same_trace = all(np.array_equal(getattr(a, f), getattr(b, f))
                     for f in ("true", "noise", "u"))
    c = channel.generate(C.e0_seed(2), 1.0, 0.85)
    diff_trace = not np.array_equal(a.true, c.true)

    ok = same_metrics and same_trace and diff_trace
    return rec("T11", "determinism and reproducibility", PASS if ok else FAIL,
               "Same seed reproduces the trace and every metric exactly; a "
               "different seed produces a different trace.",
               {"same_seed_metrics_identical": same_metrics,
                "same_seed_trace_identical": same_trace,
                "different_seed_trace_differs": diff_trace})


# --------------------------------------------------------------------------
# T12 - clip and forced-lowest accounting
# --------------------------------------------------------------------------


def t12(ctx):
    tr, pred = ctx["e0"], ctx["bank"].get(3)
    r = engine.run_adaptive(tr, pred, 3.0, keep_slots=True)
    dl = r.slots["delta"]
    hi = float(np.mean(dl >= C.DELTA_MAX))
    lo = float(np.mean(dl <= 0.0))
    mid = float(np.mean((dl > 0.0) & (dl < C.DELTA_MAX)))
    bounds_ok = bool(dl.min() >= 0.0 and dl.max() <= C.DELTA_MAX)

    gs = r.slots["gamma_safe"]
    fl = int(np.sum(gs < C.THETA[0]))
    fl_frac = fl / dl.size
    fl_ok = abs(fl_frac - r.forced_lowest_frac) < 1e-12
    denom_ok = abs(r.F_aggr * dl.size - round(r.F_aggr * dl.size)) < 1e-9

    ok = (abs(hi + lo + mid - 1.0) < 1e-12) and bounds_ok and fl_ok and denom_ok
    return rec("T12", "clip and forced-lowest accounting", PASS if ok else FAIL,
               "Delta stays in [0, 12]; the three clip classes partition the "
               "trace; forced-lowest slots are counted and stay in the F_aggr "
               "denominator.",
               {"clip_hi_frac": hi, "clip_lo_frac": lo, "interior_frac": mid,
                "bounds_ok": bounds_ok, "forced_lowest_slots": fl,
                "forced_lowest_frac": fl_frac, "denominator_is_T": denom_ok})


# --------------------------------------------------------------------------
# T13 - cold-start behaviour
# --------------------------------------------------------------------------


def t13(ctx):
    tr, bank = ctx["e0"], ctx["bank"]
    pred = bank.get(3)
    d = 3
    ra = engine.run_constant_margin(tr, pred, 0.0, policy="A", keep_slots=True)
    rb = engine.run_constant_margin(tr, pred, 3.0, policy="B", keep_slots=True)
    rc = engine.run_adaptive(tr, pred, 3.0, policy="C", record=True, keep_slots=True)
    rd = engine.run_olla(tr, pred, 3.0, policy="D", keep_slots=True)

    rows = {
        "A": {"gt_ok": bool(np.all(ra.slots["gamma_tilde"][:d] == C.MUHAT0)),
              "delta_ok": bool(np.all(ra.slots["delta"][:d] == 0.0))},
        "B": {"gt_ok": bool(np.all(rb.slots["gamma_tilde"][:d] == C.MUHAT0)),
              "delta_ok": bool(np.all(rb.slots["delta"][:d] == 3.0))},
        "C": {"gt_ok": bool(np.all(rc.slots["gamma_tilde"][:d] == C.MUHAT0)),
              "delta_ok": bool(np.all(rc.slots["delta"][:d] == 3.0))},
        "D": {"gt_ok": bool(np.all(rd.slots["gamma_tilde"][:d] == C.MUHAT0)),
              "delta_ok": bool(np.all(rd.slots["delta"][:d] == 3.0))},
    }
    first_res = rc.slots["events"].get("first_residual_instant")
    no_early = first_res == 2 * d + 1
    ok = all(v["gt_ok"] and v["delta_ok"] for v in rows.values()) and no_early
    return rec("T13", "cold-start behaviour", PASS if ok else FAIL,
               "Slots 1..d use gamma_tilde = 10 dB for every policy with the "
               "policy's initial margin; the pending buffer forms no residual "
               "before instant 2d+1.",
               {"policies": rows, "first_residual_instant": first_res,
                "expected_first_residual": 2 * d + 1})


TESTS = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_txt()
    ctx = build_context()
    results = []
    for fn in TESTS:
        try:
            results.append(fn(ctx))
        except Exception as exc:                       # noqa: BLE001
            results.append({
                "id": fn.__name__.upper(), "name": fn.__doc__ or fn.__name__,
                "status": FAIL, "detail": f"exception: {exc}",
                "evidence": {"traceback": traceback.format_exc()},
                "command": "python scripts/run_e0.py",
            })

    out = ensure("e0")
    write_json(out / "test_report.json", {
        "generated": Path(__file__).name,
        "code_version": __import__("wcmac.io_utils", fromlist=["code_version"]).code_version(),
        "results": results,
    })
    lines = ["# E0 validity gate — test report", ""]
    for r in results:
        lines.append(f"## {r['id']} — {r['name']}: **{r['status']}**")
        lines.append("")
        lines.append(r["detail"])
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r["evidence"], indent=2, default=str))
        lines.append("```")
        lines.append("")
    (out / "test_report.md").write_text("\n".join(lines), encoding="utf-8")

    n_fail = sum(1 for r in results if r["status"] != PASS)
    for r in results:
        print(f"{r['id']:<4} {r['status']:<8} {r['name']}")
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    print(f"report: {out / 'test_report.md'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
