#!/usr/bin/env python
"""Pilot gate on validation data only (Protocol §8.5; Design §12).

For every confirmatory scenario (24 main + S1-S4):
  * paired differences C - A and C - B-per on Delta_F_aggr and Delta_G
  * projected CI half-width = t_{0.975,9} * sd * kappa / sqrt(10), kappa = 1.645
  * a scenario escalates if ANY of the four projected half-widths exceeds
    max(0.20 * |effect|, floor), floors 0.025 bit/s/Hz (dG) and 0.5 pp (dF)

The H3 contrast uses the larger of the two per-cell sds at k=11 (sigma_n=1) and
k=23 (sigma_n=3), taken from the C-vs-B-per differences as a proxy because
B-nominal is a test-only arm (DEVIATIONS D-P3-01).

An Inconclusive/uncertain gate outcome is recorded as such -- it is never
written up as "no difference" or as a pass.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import config as C
from wcmac.io_utils import ensure, read_json, write_csv, write_json
from wcmac.stats import paired_ci


def load_metrics(path):
    import csv
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for key in ("G", "G_real", "BLER_obs", "F_aggr", "mean_mcs", "mean_delta",
                        "clip_hi_frac", "clip_lo_frac", "forced_lowest_frac",
                        "pred_rmse", "delta_fixed"):
                r[key] = float(r[key])
            for key in ("k", "trace_index", "seed", "d"):
                r[key] = int(r[key])
            r["sigma_n"] = float(r["sigma_n"])
            r["rho"] = float(r["rho"])
            rows.append(r)
    return rows


def main():
    rows = load_metrics(ensure("validation") / "metrics_per_run.csv")
    tuning = read_json(ensure("validation") / "tuning.json")

    by = defaultdict(dict)          # (k, policy, delta) -> {trace_index: row}
    for r in rows:
        key = (r["k"], r["policy"], round(r["delta_fixed"], 6))
        by[key][r["trace_index"]] = r

    table = []
    for k_str, t in tuning.items():
        k = int(k_str)
        dsel = round(t["delta_fixed"], 6)
        a = by[(k, "A", 0.0)]
        c = by[(k, "C", dsel)]
        b = by[(k, "B_grid", dsel)]
        idx = sorted(set(a) & set(c) & set(b))
        if len(idx) != 10:
            raise SystemExit(f"k={k}: {len(idx)} paired traces, expected 10")

        F_A = np.array([a[i]["F_aggr"] for i in idx])
        F_C = np.array([c[i]["F_aggr"] for i in idx])
        F_B = np.array([b[i]["F_aggr"] for i in idx])
        G_A = np.array([a[i]["G"] for i in idx])
        G_C = np.array([c[i]["G"] for i in idx])
        G_B = np.array([b[i]["G"] for i in idx])

        contrasts = {
            "C-A_dF": F_C - F_A, "C-A_dG": G_C - G_A,
            "C-B_dF": F_C - F_B, "C-B_dG": G_C - G_B,
        }
        entry = {"k": k, "sid": t["sid"], "sigma_n": t["sigma_n"], "rho": t["rho"],
                 "d": t["d"], "delta_fixed": t["delta_fixed"],
                 "F_star": t["F_star"], "contrasts": {}}
        triggered = False
        for name, diff in contrasts.items():
            ci = paired_ci(diff, np.zeros_like(diff))
            metric = name.split("_")[1]
            floor = C.GATE_FLOOR_DG if metric == "dG" else C.GATE_FLOOR_DF
            half = ci.half * C.KAPPA
            thr = max(C.GATE_EFFECT_FRACTION * abs(ci.mean), floor)
            trig = bool(half > thr)
            triggered |= trig
            entry["contrasts"][name] = {
                "mean": ci.mean, "sd": ci.sd, "half_raw": ci.half,
                "half_kappa": half, "threshold": thr, "trigger": trig,
                "floor": floor,
            }
        entry["escalate"] = triggered if t["split"] == "main" or C.SCENARIOS[k].confirmatory else False
        entry["confirmatory"] = bool(C.SCENARIOS[k].confirmatory)
        table.append(entry)

    # --- H3 proxy: larger of the two per-cell sds at k=11 and k=23 ---
    by_k = {e["k"]: e for e in table}
    sd11 = by_k[11]["contrasts"]["C-B_dF"]["sd"]
    sd23 = by_k[23]["contrasts"]["C-B_dF"]["sd"]
    sd_h3 = max(sd11, sd23)
    half_h3 = C.KAPPA * 2.2621571627982 * sd_h3 / np.sqrt(10)   # t_{0.975,9}
    thr_h3 = max(C.GATE_EFFECT_FRACTION * abs(by_k[23]["contrasts"]["C-B_dF"]["mean"]),
                 C.GATE_FLOOR_DF)
    h3_trigger = bool(half_h3 > thr_h3)

    escalated = sorted(e["k"] for e in table if e["escalate"])
    n_main_esc = sum(1 for e in table if e["escalate"] and e["k"] in C.MAIN_KS)
    n_stress_esc = sum(1 for e in table if e["escalate"] and e["k"] in C.STRESS_KS)
    extra_runs = n_main_esc * 40 + n_stress_esc * 30

    out = ensure("gate")
    write_json(out / "pilot_gate.json", {
        "kappa": C.KAPPA, "floors": {"dG": C.GATE_FLOOR_DG, "dF": C.GATE_FLOOR_DF},
        "scenarios": table,
        "h3_proxy": {"sd_k11": sd11, "sd_k23": sd23, "sd_used": sd_h3,
                     "half_kappa": half_h3, "threshold": thr_h3,
                     "trigger": h3_trigger,
                     "note": "C-vs-B-per proxy; B-nominal is a test-only arm (D-P3-01)"},
        "escalated": escalated,
        "n_main_escalated": n_main_esc,
        "n_stress_escalated": n_stress_esc,
        "extra_test_traces_runs": extra_runs,
        "e5_s5_not_gated": True,
    })

    flat = []
    for e in table:
        r = {"k": e["k"], "sid": e["sid"], "sigma_n": e["sigma_n"], "rho": e["rho"],
             "d": e["d"], "delta_fixed": e["delta_fixed"], "F_star": e["F_star"],
             "confirmatory": e["confirmatory"], "escalate": e["escalate"]}
        for name, c in e["contrasts"].items():
            for f in ("mean", "sd", "half_raw", "half_kappa", "threshold", "trigger"):
                r[f"{name}_{f}"] = c[f]
        flat.append(r)
    write_csv(out / "pilot_gate.csv", flat)

    lines = ["# Pilot gate — validation data only", "",
             f"kappa = {C.KAPPA:.4f}; floors dG = {C.GATE_FLOOR_DG} bit/s/Hz, "
             f"dF = {C.GATE_FLOOR_DF:.4f} (0.5 pp)", "",
             "| k | sid | s_n | rho | d | D* | C-A dF hw/thr | C-A dG hw/thr | "
             "C-B dF hw/thr | C-B dG hw/thr | escalate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in table:
        def cell(n):
            c = e["contrasts"][n]
            return f"{c['half_kappa']:.4f} / {c['threshold']:.4f}{' **' if c['trigger'] else ''}"
        lines.append(f"| {e['k']} | {e['sid']} | {e['sigma_n']} | {e['rho']} | {e['d']} | "
                     f"{e['delta_fixed']:.2f} | {cell('C-A_dF')} | {cell('C-A_dG')} | "
                     f"{cell('C-B_dF')} | {cell('C-B_dG')} | {'YES' if e['escalate'] else 'no'} |")
    lines += ["", f"H3 proxy sd (max of k=11, k=23) = {sd_h3:.4f}; "
                  f"projected half-width = {half_h3:.4f} vs threshold {thr_h3:.4f}; "
                  f"trigger = {h3_trigger}",
              "", f"Escalated: {escalated}",
              f"Extra test policy-runs: 24-matrix {n_main_esc}x40 + stress {n_stress_esc}x30 "
              f"= {extra_runs}", ""]
    (out / "pilot_gate.md").write_text("\n".join(lines), encoding="utf-8")

    for e in table:
        marks = "".join("*" if c["trigger"] else "." for c in e["contrasts"].values())
        print(f"k={e['k']:02d} {e['sid']:>3} {marks} {'ESCALATE' if e['escalate'] else ''}")
    print(f"\nH3 proxy: sd={sd_h3:.4f} hw={half_h3:.4f} thr={thr_h3:.4f} trigger={h3_trigger}")
    print(f"escalated {len(escalated)} scenarios -> +{extra_runs} test runs")
    print(f"report: {out / 'pilot_gate.md'}")


if __name__ == "__main__":
    main()
