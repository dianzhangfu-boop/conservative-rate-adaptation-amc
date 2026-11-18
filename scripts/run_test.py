#!/usr/bin/env python
"""Confirmatory test runs. Refuses to execute unless the artefact is frozen.

Test traces are generated once, archived, and checksummed; every later analysis
reads the archive (Protocol §2.2 rule 5).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import channel, config as C, engine, metrics as M, predictor as P
from wcmac.io_utils import (ART, code_version, ensure, read_json, sha256_file,
                            write_csv, write_json)


def require_frozen():
    lock = ART / "FROZEN.json"
    if not lock.exists():
        raise SystemExit("refusing to run: artefact is not frozen (run scripts/freeze.py)")
    frozen = read_json(lock)
    now = code_version()
    if now != frozen["code_version"]:
        raise SystemExit(
            "refusing to run: code changed after freeze\n"
            f"  frozen {frozen['code_version']}\n  now    {now}\n"
            "Any code change after the freeze invalidates the test-set guarantee."
        )
    return frozen


def row(sc, i, seed, policy, delta, res, prmse, extra=None):
    r = {"split": "test", "k": sc.k, "sid": sc.sid, "sigma_n": sc.sigma_n,
         "rho": sc.rho, "d": sc.d, "trace_index": i, "seed": seed,
         "policy": policy, "delta_fixed": delta,
         "G": res.G, "G_real": res.G_real, "BLER_obs": res.BLER_obs,
         "F_aggr": res.F_aggr, "mean_mcs": res.mean_mcs,
         "mean_delta": res.mean_delta, "clip_hi_frac": res.clip_hi_frac,
         "clip_lo_frac": res.clip_lo_frac,
         "forced_lowest_frac": res.forced_lowest_frac, "pred_rmse": prmse}
    if extra:
        r.update(extra)
    return r


def archive(traces_by_key, path):
    """Persist test traces and their checksums; never regenerate in analysis."""
    arrays = {}
    manifest = {}
    for key, tr in traces_by_key.items():
        arrays[f"{key}_true"] = tr.true
        arrays[f"{key}_noise"] = tr.noise
        arrays[f"{key}_u"] = tr.u
        manifest[key] = {
            "seed": tr.seed,
            "sha_true": sha256_file_bytes(tr.true), "sha_noise": sha256_file_bytes(tr.noise),
            "sha_u": sha256_file_bytes(tr.u), "n": int(tr.true.size),
        }
    np.savez_compressed(path, **arrays)
    write_json(str(path) + ".manifest.json", manifest)
    return manifest


def sha256_file_bytes(a) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    frozen = require_frozen()
    import csv
    out = ensure("test")
    t0 = time.time()

    bank = P.PredictorBank({int(d): P.Predictor(**{k: v for k, v in p.items() if k != "d"},
                                               d=int(d))
                            for d, p in frozen["predictors"].items()})
    delta_fixed = {int(k): v for k, v in frozen["delta_fixed"].items()}
    delta_nominal = delta_fixed[C.NOMINAL_K]
    escalated = set(frozen["escalation"]["escalated_scenarios"])

    rows, traces = [], {}
    n_runs = 0

    # ---------------- main matrix ----------------
    for k in C.MAIN_KS:
        sc = C.SCENARIOS[k]
        pred = bank.get(sc.d)
        n_tr = C.ESCALATED_TRACES_PER_SCENARIO if k in escalated else C.TRACES_PER_SCENARIO
        for i in range(1, n_tr + 1):
            seed = C.main_seed(k, i)
            tr = channel.generate(seed, sc.sigma_n, sc.rho, scenario=sc.sid, split="main_test")
            traces[f"k{k}_{i}"] = tr
            prmse = float(np.sqrt(np.mean((engine._predictions(tr, pred)[sc.d:] - tr.true[sc.d:]) ** 2)))
            for pol, delta, extra in (
                ("A", 0.0, None),
                ("B_per", delta_fixed[k], None),
                ("B_nom", delta_nominal, None),
                ("C", delta_fixed[k], {"delta_init": delta_fixed[k]}),
            ):
                r = (engine.run_adaptive(tr, pred, delta_fixed[k], policy="C")
                     if pol == "C" else engine.run_constant_margin(tr, pred, delta, policy=pol))
                rows.append(row(sc, i, seed, pol, delta, r, prmse, extra))
                n_runs += 1
        print(f"main k={k:02d} {sc.sid} done ({n_runs} runs)")

    # ---------------- stress matrix ----------------
    for k in C.STRESS_KS:
        sc = C.SCENARIOS[k]
        pred = bank.get(sc.d)
        n_tr = C.ESCALATED_TRACES_PER_SCENARIO if k in escalated else C.TRACES_PER_SCENARIO
        for i in range(1, n_tr + 1):
            seed = C.stress_seed(k, i)
            tr = channel.generate(seed, sc.sigma_n, sc.rho, scenario=sc.sid, split="stress_test")
            traces[f"k{k}_{i}"] = tr
            prmse = float(np.sqrt(np.mean((engine._predictions(tr, pred)[sc.d:] - tr.true[sc.d:]) ** 2)))
            for pol, delta in (("A", 0.0), ("B_per", delta_fixed[k]), ("C", delta_fixed[k])):
                r = (engine.run_adaptive(tr, pred, delta_fixed[k], policy="C")
                     if pol == "C" else engine.run_constant_margin(tr, pred, delta, policy=pol))
                rows.append(row(sc, i, seed, pol, delta, r, prmse,
                                {"delta_init": delta} if pol == "C" else None))
                n_runs += 1
        print(f"stress k={k:02d} {sc.sid} done ({n_runs} runs)")

    # ---------------- E5 (H5, secondary) ----------------
    sc5 = C.SCENARIOS[C.NOMINAL_K]
    pred = bank.get(sc5.d)
    d5 = delta_fixed[C.NOMINAL_K]
    for i in range(11, 21):
        seed = C.e5_seed(i)
        tr = channel.generate(seed, 1.0, 0.85, mu_step=True, scenario="E5", split="e5_test")
        traces[f"e5_{i}"] = tr
        prmse = float(np.sqrt(np.mean((engine._predictions(tr, pred)[sc5.d:] - tr.true[sc5.d:]) ** 2)))
        for pol in ("C", "D"):
            r = (engine.run_adaptive(tr, pred, d5, policy="C", keep_slots=True) if pol == "C"
                 else engine.run_olla(tr, pred, d5, policy="D", keep_slots=True))
            seg = M.segment_means(r.slots["aggr"])
            tau, censored = M.recovery_time(r.slots["aggr"])
            rows.append(row(sc5, i, seed, pol, d5, r, prmse,
                            {"delta_init": d5, "e5": True,
                             "F_pre": seg["F_pre"], "F_trans": seg["F_trans"],
                             "F_post": seg["F_post"], "tau": tau,
                             "censored": int(censored)}))
            n_runs += 1
    print(f"E5 done ({n_runs} runs)")

    # ---------------- archive + report ----------------
    manifest = archive(traces, out / "test_traces.npz")
    write_csv(out / "metrics_per_run.csv", rows)
    write_json(out / "runlog.json", {
        "n_runs": n_runs, "n_traces": len(traces), "wall_seconds": time.time() - t0,
        "metrics_sha256": sha256_file(out / "metrics_per_run.csv"),
        "traces_npz_sha256": sha256_file(out / "test_traces.npz"),
        "n_trace_checksums": len(manifest),
        "escalated": sorted(escalated),
        "code_version": frozen["code_version"],
    })
    print(f"\ntest policy-runs: {n_runs}   traces: {len(traces)}   "
          f"wall={time.time() - t0:.1f}s")
    print(f"archive: {out / 'test_traces.npz'} ({sha256_file(out / 'test_traces.npz')[:16]})")


if __name__ == "__main__":
    main()
