#!/usr/bin/env python
"""Descriptive experiments E6 and E7 on the archived nominal test traces.

Descriptive only: alpha = 0.90 and N_w = 100 remain the confirmatory settings
regardless of what these runs show (Protocol §2.2 rule 7, §11.4).  Nothing here
feeds back into any hypothesis decision.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))      # `-I` drops the script directory

from wcmac import channel, config as C, engine, predictor as P
from wcmac.io_utils import ART, code_version, ensure, read_json, write_csv, write_json

from run_test import require_frozen


def load_nominal(frozen, k=C.NOMINAL_K):
    sc = C.SCENARIOS[k]
    pred = P.PredictorBank({int(d): P.Predictor(**{kk: vv for kk, vv in p.items() if kk != "d"},
                                               d=int(d))
                            for d, p in frozen["predictors"].items()}).get(sc.d)
    traces = [channel.generate(C.main_seed(k, i), sc.sigma_n, sc.rho,
                               scenario=sc.sid, split="main_test") for i in range(1, 11)]
    return sc, pred, traces


def main():
    frozen = require_frozen()
    out = ensure("descriptive")
    t0 = time.time()
    sc, pred, traces = load_nominal(frozen)
    delta_fixed = frozen["delta_fixed"][str(C.NOMINAL_K)]
    rows = []

    # ---------------- E6: N_w x alpha ----------------
    for n_w in (10, 25, 50, 100, 200, 500):
        for alpha in (0.80, 0.90, 0.95, 0.99):
            opt = C.Options(n_w=n_w, alpha=alpha)
            for i, tr in enumerate(traces, start=1):
                r = engine.run_adaptive(tr, pred, delta_fixed, opt=opt, policy="C")
                rows.append({"experiment": "E6", "config": f"Nw{n_w}_a{alpha:.2f}",
                             "n_w": n_w, "alpha": alpha, "policy": "C",
                             "trace_index": i, "G": r.G, "F_aggr": r.F_aggr,
                             "BLER_obs": r.BLER_obs, "mean_delta": r.mean_delta,
                             "clip_hi_frac": r.clip_hi_frac,
                             "clip_lo_frac": r.clip_lo_frac})
    print(f"E6 done ({len(rows)} runs)")

    # ---------------- E7: BLER-curve sensitivity ----------------
    configs = []
    for a in (0.7, 1.0, 1.4):
        configs.append((f"S1_a{a}", C.Options(a_slope=(a, a, a, a))))
    for shift in (-1.0, 1.0):
        configs.append((f"S2_b{shift:+.0f}", C.Options(b_mid=tuple(b + shift for b in C.B_MID))))
    configs.append(("S3_sp3", C.Options(b_mid=(0.0, 3.0, 6.0, 9.0))))
    configs.append(("S3_sp5", C.Options(b_mid=(0.0, 5.0, 10.0, 15.0))))
    for beta in (0.05, 0.10, 0.20):
        configs.append((f"S4_beta{beta:.2f}", C.Options(beta=beta)))

    for name, opt in configs:
        for pol in ("A", "B", "C"):
            for i, tr in enumerate(traces, start=1):
                if pol == "C":
                    r = engine.run_adaptive(tr, pred, delta_fixed, opt=opt, policy="C")
                else:
                    r = engine.run_constant_margin(tr, pred, 0.0 if pol == "A" else delta_fixed,
                                                   opt=opt, policy=pol)
                rows.append({"experiment": "E7", "config": name, "n_w": C.N_W,
                             "alpha": C.ALPHA, "policy": pol, "trace_index": i,
                             "G": r.G, "F_aggr": r.F_aggr, "BLER_obs": r.BLER_obs,
                             "mean_delta": r.mean_delta,
                             "clip_hi_frac": r.clip_hi_frac,
                             "clip_lo_frac": r.clip_lo_frac})
    print(f"E6+E7 done ({len(rows)} runs total)")

    write_csv(out / "descriptive_runs.csv", rows)
    write_json(out / "descriptive_runlog.json", {
        "n_runs": len(rows), "wall_seconds": time.time() - t0,
        "code_version": frozen["code_version"],
        "status": "descriptive only; not used for any hypothesis decision",
    })
    print(f"wall={time.time() - t0:.1f}s  -> {out / 'descriptive_runs.csv'}")


if __name__ == "__main__":
    main()
