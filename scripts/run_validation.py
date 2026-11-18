#!/usr/bin/env python
"""Fit the predictor (training split) and run the validation phase.

Order fixed by the protocol:
  1. fit (c0^(d), c1^(d)) on the 30 training traces, all four delays;
  2. Policy C on the 10 validation traces of every scenario  -> F*_k
  3. Policy A on the same traces                            -> gate contrast
  4. monotone binary search over the 0.25 dB grid for Delta_fixed(k)
  5. re-derive C's validation metrics with the now-locked Delta_init = Delta_fixed

Step 5 is exact and costs no extra runs: after the window reaches n_min the
margin trajectory is a function of the measurements alone, so changing
Delta_init alters only the warm-up slots, which are recomputed from the stored
path (DEVIATIONS D-P3-06).

Writes
    artefact/predictor.json
    artefact/validation/metrics_per_run.csv
    artefact/validation/tuning.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import channel, config as C, engine, predictor as P
from wcmac.io_utils import ensure, env_txt, sha256_file, write_csv, write_json
from wcmac.metrics import prediction_rmse

PROVISIONAL_DELTA_INIT = 3.0


def fit_predictor() -> P.PredictorBank:
    train, ids, j = [], [], 0
    for (rho, sn), n in C.TRAINING_POOL.items():
        for _ in range(n):
            j += 1
            train.append(channel.generate_train(C.train_seed(j), sn, rho))
            ids.append(f"train:{j}")
    ledger = []
    bank = P.PredictorBank({d: P.fit(train, d, ids, ledger) for d in C.TRAIN_DS})
    write_json(ensure() / "predictor.json", {
        "predictors": bank.as_dict(),
        "training_pool": {f"rho={k[0]},sigma_n={k[1]}": v for k, v in C.TRAINING_POOL.items()},
        "training_seeds": [C.train_seed(j) for j in range(1, 31)],
        "ledger": ledger,
    })
    return bank


def load_traces(sc, which: str) -> list:
    fn = sc.val_seed_fn if which == "val" else sc.test_seed_fn
    return [channel.generate(fn(sc.k, i), sc.sigma_n, sc.rho,
                             scenario=sc.sid, split=f"{sc.split}_{which}")
            for i in range(1, C.TRACES_PER_SCENARIO + 1)]


def pred_rmse(tr, pred):
    return prediction_rmse(engine._predictions(tr, pred), tr.true, pred.d)


def row(sc, split, i, seed, policy, delta, res, prmse, extra=None):
    r = {
        "split": split, "k": sc.k, "sid": sc.sid, "sigma_n": sc.sigma_n,
        "rho": sc.rho, "d": sc.d, "trace_index": i, "seed": seed,
        "policy": policy, "delta_fixed": delta,
        "G": res.G, "G_real": res.G_real, "BLER_obs": res.BLER_obs,
        "F_aggr": res.F_aggr, "mean_mcs": res.mean_mcs,
        "mean_delta": res.mean_delta, "clip_hi_frac": res.clip_hi_frac,
        "clip_lo_frac": res.clip_lo_frac, "forced_lowest_frac": res.forced_lowest_frac,
        "pred_rmse": prmse,
    }
    if extra:
        r.update(extra)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    env_txt()
    t0 = time.time()
    out = ensure("validation")
    bank = fit_predictor()
    print("predictor:", {d: (round(p.c0, 4), round(p.c1, 4)) for d, p in bank.by_d.items()})

    wanted = {int(x) for x in args.only.split(",")} if args.only else None
    rows, tuning = [], {}
    n_runs = 0

    for k in sorted(C.SCENARIOS):
        sc = C.SCENARIOS[k]
        if wanted and k not in wanted:
            continue
        pred = bank.get(sc.d)
        traces = load_traces(sc, "val")
        prmse = float(np.mean([pred_rmse(t, pred) for t in traces]))

        # --- Policy A on validation (pilot-gate contrast; also supplies the
        #     Delta = 0 grid point exactly, by A == B(0)) ---
        a_faggr = []
        for i, tr in enumerate(traces, start=1):
            r = engine.run_constant_margin(tr, pred, 0.0, policy="A")
            a_faggr.append(r.F_aggr)
            rows.append(row(sc, "val", i, tr.seed, "A", 0.0, r, prmse))
            n_runs += 1

        # --- Policy C, provisional Delta_init, keep the path for step 5 ---
        c_res, c_preds = [], []
        c_paths = []
        for i, tr in enumerate(traces, start=1):
            r = engine.run_adaptive(tr, pred, PROVISIONAL_DELTA_INIT, policy="C",
                                    keep_slots=True, record=True)
            c_res.append(r)
            c_paths.append((r.slots["gamma_tilde"], r.slots["delta"].copy(),
                            r.slots["events"]["handover_instant"]))
            c_preds.append(prmse)
            n_runs += 1
        fstar = float(np.mean([r.F_aggr for r in c_res]))

        # --- Delta_fixed grid search ---
        grid = C.DELTA_GRID
        cache: dict[int, float] = {}

        def F(j: int) -> float:
            nonlocal n_runs
            if j == 0:                       # A == B(0): reuse the archived run
                cache[0] = float(np.mean(a_faggr))
                return cache[0]
            if j not in cache:
                vals = []
                for i, tr in enumerate(traces, start=1):
                    r = engine.run_constant_margin(tr, pred, grid[j], policy="B_grid")
                    vals.append(r.F_aggr)
                    rows.append(row(sc, "val", i, tr.seed, "B_grid", grid[j], r, prmse))
                    n_runs += 1
                cache[j] = float(np.mean(vals))
            return cache[j]

        hi = len(grid) - 1
        matched = F(hi) <= fstar
        if not matched:
            delta_fixed = grid[hi]          # best attainable; matching failed
        else:
            lo = 0
            while lo < hi:
                mid = (lo + hi) // 2
                if F(mid) <= fstar:
                    hi = mid
                else:
                    lo = mid + 1
            delta_fixed = grid[lo]

        # --- Step 5: C metrics with the locked Delta_init = Delta_fixed ---
        for i, (tr, (gt, dl, hand)) in enumerate(zip(traces, c_paths), start=1):
            dl2 = dl.copy()
            dl2[: hand - 1] = delta_fixed
            r2 = engine.result_from_path(tr, gt, dl2, policy="C")
            rows.append(row(sc, "val", i, tr.seed, "C", delta_fixed, r2, prmse,
                            extra={"delta_init": delta_fixed,
                                   "F_aggr_provisional": c_res[i - 1].F_aggr}))
            c_res[i - 1] = r2

        fstar_locked = float(np.mean([r.F_aggr for r in c_res]))
        js = sorted(cache)
        tuning[k] = {
            "sid": sc.sid, "sigma_n": sc.sigma_n, "rho": sc.rho, "d": sc.d,
            "split": sc.split,
            "F_star_provisional": fstar,
            "F_star": fstar_locked,
            "delta_fixed": delta_fixed,
            "matched": bool(matched),
            "grid_evaluations": {str(grid[j]): cache[j] for j in js},
            "n_grid_points_evaluated": len(cache),
            "monotone_in_cache": bool(all(cache[a] >= cache[b] - 1e-12
                                          for a, b in zip(js, js[1:]))),
            "pred_rmse_mean": prmse,
        }
        print(f"k={k:02d} {sc.sid:>3} F*={fstar_locked:.4f} D*={delta_fixed:5.2f} "
              f"matched={matched} pts={len(cache)} rmse={prmse:.2f}")

    write_csv(out / "metrics_per_run.csv", rows)
    write_json(out / "tuning.json", tuning)
    write_json(ensure() / "validation_runlog.json", {
        "n_policy_runs": n_runs,
        "wall_seconds": time.time() - t0,
        "metrics_sha256": sha256_file(out / "metrics_per_run.csv"),
        "provisional_delta_init": PROVISIONAL_DELTA_INIT,
    })
    print(f"\nvalidation policy-runs: {n_runs}   wall={time.time() - t0:.1f}s")
    print(f"metrics: {out / 'metrics_per_run.csv'}")


if __name__ == "__main__":
    main()
