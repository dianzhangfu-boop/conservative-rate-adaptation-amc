#!/usr/bin/env python
"""Pre-registered analysis of the confirmatory test runs (H1-H5).

Reads only artefact/test/metrics_per_run.csv (produced from the archived
traces).  Every decision uses the three-state CI rules of Design §3; a
non-significant result is Inconclusive, never Contradicted and never read as
equivalence.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import config as C
from wcmac.io_utils import ensure, read_json, write_csv, write_json
from wcmac.stats import (CONTRADICTED, INCONCLUSIVE, SUPPORTED, class_at_most,
                         class_inside, class_lower, class_upper, holm, joint,
                         paired_ci, welch_ci)

FLOAT_FIELDS = ("G", "G_real", "BLER_obs", "F_aggr", "mean_mcs", "mean_delta",
                "clip_hi_frac", "clip_lo_frac", "forced_lowest_frac", "pred_rmse",
                "delta_fixed", "delta_init", "F_pre", "F_trans", "F_post", "tau")
INT_FIELDS = ("k", "trace_index", "seed", "d", "censored")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for f in FLOAT_FIELDS:
                if f in r and r[f] not in ("", None):
                    try:
                        r[f] = float(r[f])
                    except ValueError:
                        pass
            for f in INT_FIELDS:
                if f in r and r[f] not in ("", None):
                    r[f] = int(float(r[f]))
            rows.append(r)
    return rows


def cells(rows):
    by = defaultdict(dict)
    for r in rows:
        by[(r["k"], r["policy"])][r["trace_index"]] = r
    return by


def pair(by, k, pa, pb, field):
    a, b = by[(k, pa)], by[(k, pb)]
    idx = sorted(set(a) & set(b))
    return (np.array([a[i][field] for i in idx]),
            np.array([b[i][field] for i in idx]), idx)


def mde(sd, n, level=0.95):
    df = n - 1
    return (float(sps.t.ppf(0.5 + level / 2, df)) + float(sps.t.ppf(0.80, df))) * sd / np.sqrt(n)


def main():
    rows = load(ensure("test") / "metrics_per_run.csv")
    by = cells(rows)
    frozen = read_json(ensure() / "FROZEN.json")
    out = ensure("analysis")
    H = {}

    def ci_dict(ci):
        return ci.as_dict()

    def table_row(k, name, ci, state, extra=None):
        r = {"k": k, "contrast": name, "n": ci.n, "mean": ci.mean, "sd": ci.sd,
             "ci_lo": ci.lo, "ci_hi": ci.hi, "t": ci.t, "p": ci.p, "dz": ci.dz,
             "mde": mde(ci.sd, ci.n), "state": state}
        if extra:
            r.update(extra)
        return r

    # ------------------------------------------------------------------ H1
    h1_rows, p_rq1 = [], {"dF": {}, "dB": {}, "dG": {}}
    for k in C.MAIN_KS:
        cF, aF, _ = pair(by, k, "C", "A", "F_aggr")
        cB, aB, _ = pair(by, k, "C", "A", "BLER_obs")
        cG, aG, _ = pair(by, k, "C", "A", "G")
        ciF, ciB, ciG = paired_ci(cF, aF), paired_ci(cB, aB), paired_ci(cG, aG)
        sF, sB, sG = class_lower(ciF.lo, ciF.hi), class_lower(ciB.lo, ciB.hi), class_lower(ciG.lo, ciG.hi)
        state = joint([sF, sB])
        h1_rows.append(table_row(k, "C-A", ciF, sF, {"metric": "F_aggr"}))
        h1_rows.append(table_row(k, "C-A", ciB, sB, {"metric": "BLER_obs"}))
        h1_rows.append(table_row(k, "C-A", ciG, sG, {"metric": "G", "decision": False}))
        p_rq1["dF"][k] = ciF.p
        p_rq1["dB"][k] = ciB.p
        p_rq1["dG"][k] = ciG.p
        H.setdefault("H1", {})[k] = {"state": state, "leg_F": sF, "leg_BLER": sB,
                                    "leg_G_descriptive": sG,
                                    "dF": ci_dict(ciF), "dBLER": ci_dict(ciB),
                                    "dG": ci_dict(ciG)}
    H["H1_summary"] = {
        "supported": sum(1 for v in H["H1"].values() if v["state"] == SUPPORTED),
        "contradicted": sum(1 for v in H["H1"].values() if v["state"] == CONTRADICTED),
        "inconclusive": sum(1 for v in H["H1"].values() if v["state"] == INCONCLUSIVE),
        "n_scenarios": len(H["H1"]),
        "holm_RQ1": {m: holm(v) for m, v in p_rq1.items()},
    }

    # ------------------------------------------------------------------ H2
    h2_rows, p_rq2 = [], {}
    for k in C.MAIN_KS:
        cG, bG, _ = pair(by, k, "C", "B_per", "G")
        cF, bF, _ = pair(by, k, "C", "B_per", "F_aggr")
        ciG, ciF = paired_ci(cG, bG), paired_ci(cF, bF)
        sG = class_upper(ciG.lo, ciG.hi)
        sF = class_at_most(ciF.lo, ciF.hi, 0.01)
        state = joint([sG, sF])
        h2_rows.append(table_row(k, "C-Bper", ciG, sG, {"metric": "G"}))
        h2_rows.append(table_row(k, "C-Bper", ciF, sF, {"metric": "F_aggr"}))
        p_rq2[k] = ciG.p
        H.setdefault("H2", {})[k] = {"state": state, "leg_G": sG, "leg_F": sF,
                                    "dG": ci_dict(ciG), "dF": ci_dict(ciF)}
    H["H2_summary"] = {
        "supported": sum(1 for v in H["H2"].values() if v["state"] == SUPPORTED),
        "contradicted": sum(1 for v in H["H2"].values() if v["state"] == CONTRADICTED),
        "inconclusive": sum(1 for v in H["H2"].values() if v["state"] == INCONCLUSIVE),
        "n_scenarios": len(H["H2"]),
        "holm_RQ2": holm(p_rq2),
    }

    # ------------------------------------------------------------------ H3
    cells_h3 = {0.0: 5, 1.0: 11, 2.0: 17, 3.0: 23}
    D, Dci = {}, {}
    for sn, k in cells_h3.items():
        cF, bF, _ = pair(by, k, "C", "B_nom", "F_aggr")
        D[sn] = cF - bF
        Dci[sn] = paired_ci(cF, bF)
    # welch_ci takes the raw (a,b) pairs of each cell, not the differences
    cF1, bF1, _ = pair(by, 11, "C", "B_nom", "F_aggr")
    cF3, bF3, _ = pair(by, 23, "C", "B_nom", "F_aggr")
    contrast = welch_ci(cF1, bF1, cF3, bF3, df_fixed=9)
    s_contrast = class_lower(contrast.lo, contrast.hi)
    s_level = class_lower(Dci[3.0].lo, Dci[3.0].hi)
    H["H3"] = {
        "state": joint([s_contrast, s_level]),
        "leg_contrast": s_contrast, "leg_level": s_level,
        "D_by_sigma": {str(s): ci_dict(Dci[s]) for s in Dci},
        "contrast_D3_minus_D1": ci_dict(contrast),
    }

    # ------------------------------------------------------------------ H4
    h4_rows, per_cell = [], {}
    for k in (25, 26, 27, 28):
        cG, bG, _ = pair(by, k, "C", "B_per", "G")
        cF, bF, _ = pair(by, k, "C", "B_per", "F_aggr")
        ciG, ciF = paired_ci(cG, bG), paired_ci(cF, bF)
        if ciG.hi <= 0.05 or ciF.lo > 0.01:
            state = SUPPORTED          # H4's claim holds: C fails to beat B
            kind = "failure"
        elif ciG.lo > 0.05 and ciF.hi <= 0.01:
            state = CONTRADICTED       # demonstrated robustness
            kind = "robust"
        else:
            state = INCONCLUSIVE
            kind = "inconclusive"
        per_cell[k] = {"state": state, "kind": kind,
                       "dG": ci_dict(ciG), "dF": ci_dict(ciF)}
        h4_rows.append(table_row(k, "C-Bper", ciG, state, {"metric": "G", "cell_kind": kind}))
        h4_rows.append(table_row(k, "C-Bper", ciF, state, {"metric": "F_aggr", "cell_kind": kind}))
    f = sum(1 for v in per_cell.values() if v["kind"] == "failure")
    r = sum(1 for v in per_cell.values() if v["kind"] == "robust")
    if f >= 2 and r == 0:
        overall = SUPPORTED
    elif r == 4:
        overall = CONTRADICTED
    else:
        overall = INCONCLUSIVE
    H["H4"] = {"cells": per_cell, "f": f, "r": r, "state": overall}

    # ------------------------------------------------------------------ H5
    e5 = [r for r in rows if r.get("e5")]
    if e5:
        def e5pair(field):
            byp = defaultdict(dict)
            for r in e5:
                byp[r["policy"]][r["trace_index"]] = r
            idx = sorted(set(byp["C"]) & set(byp["D"]))
            return (np.array([byp["C"][i][field] for i in idx]),
                    np.array([byp["D"][i][field] for i in idx]), idx)

        preC, preD, _ = e5pair("F_pre")
        trC, trD, _ = e5pair("F_trans")
        tauC, tauD, idx = e5pair("tau")
        byp = defaultdict(dict)
        for r in e5:
            byp[r["policy"]][r["trace_index"]] = r
        censC = np.array([byp["C"][i]["censored"] for i in idx])
        censD = np.array([byp["D"][i]["censored"] for i in idx])

        ci_pre = paired_ci(preC, preD)
        ci_trans = paired_ci(trC, trD)
        ci_tau = paired_ci(tauC, tauD)

        s_pre = class_inside(ci_pre.lo, ci_pre.hi, 0.02)
        if np.all(censC & censD):
            s_tau = INCONCLUSIVE
        else:
            s_tau = class_lower(ci_tau.lo, ci_tau.hi)
        s_trans = class_lower(ci_trans.lo, ci_trans.hi)
        H["H5"] = {
            "state": joint([s_pre, s_tau, s_trans]),
            "leg_pre": s_pre, "leg_tau": s_tau, "leg_trans": s_trans,
            "dF_pre": ci_dict(ci_pre), "dTau": ci_dict(ci_tau),
            "dF_trans": ci_dict(ci_trans),
            "censored_C": int(censC.sum()), "censored_D": int(censD.sum()),
            "n_pairs": int(ci_tau.n),
        }
    else:
        H["H5"] = {"state": "NOT RUN"}

    # ------------------------------------------------------------------ write
    write_json(out / "hypotheses.json", H)
    write_csv(out / "h1_contrasts.csv", h1_rows)
    write_csv(out / "h2_contrasts.csv", h2_rows)
    write_csv(out / "h4_contrasts.csv", h4_rows)

    lines = ["# Confirmatory analysis", ""]
    lines += ["## H1 — C vs A (primary)", "",
              f"Supported {H['H1_summary']['supported']}/24, "
              f"Contradicted {H['H1_summary']['contradicted']}/24, "
              f"Inconclusive {H['H1_summary']['inconclusive']}/24", "",
              "| k | dF_aggr [CI] | state | dBLER [CI] | state | dG (desc) | state |",
              "|---|---|---|---|---|---|---|"]
    for k in C.MAIN_KS:
        v = H["H1"][k]
        d = v["dF"]; b = v["dBLER"]; g = v["dG"]
        lines.append(f"| {k} | {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] | {v['leg_F']} | "
                     f"{b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] | {v['leg_BLER']} | "
                     f"{g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] | {v['leg_G_descriptive']} |")
    lines += ["", "## H2 — C vs B-per-scenario (matched reliability)", "",
              f"Supported {H['H2_summary']['supported']}/24, "
              f"Contradicted {H['H2_summary']['contradicted']}/24, "
              f"Inconclusive {H['H2_summary']['inconclusive']}/24", "",
              "| k | dG [CI] | leg | dF [CI] vs +1pp | leg | state |",
              "|---|---|---|---|---|---|"]
    for k in C.MAIN_KS:
        v = H["H2"][k]
        g = v["dG"]; f_ = v["dF"]
        lines.append(f"| {k} | {g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] | {v['leg_G']} | "
                     f"{f_['mean']:+.4f} [{f_['lo']:+.4f},{f_['hi']:+.4f}] | {v['leg_F']} | {v['state']} |")
    h3 = H["H3"]
    lines += ["", "## H3 — noise robustness (C vs B-nominal)", "",
              f"state: **{h3['state']}** (contrast leg {h3['leg_contrast']}, level leg {h3['leg_level']})", "",
              "| sigma_n | D = F_C - F_Bnom [CI] |", "|---|---|"]
    for s, v in h3["D_by_sigma"].items():
        lines.append(f"| {s} | {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] |")
    c = h3["contrast_D3_minus_D1"]
    lines.append(f"| D(3)-D(1) | {c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] |")
    h4 = H["H4"]
    lines += ["", "## H4 — stress matrix", "",
              f"failure cells f = {h4['f']}, robust cells r = {h4['r']} → **{h4['state']}**", "",
              "| k | dG [CI] | dF [CI] | cell |", "|---|---|---|---|"]
    for k, v in h4["cells"].items():
        g, f_ = v["dG"], v["dF"]
        lines.append(f"| {k} | {g['mean']:+.4f} [{g['lo']:+.4f},{g['hi']:+.4f}] | "
                     f"{f_['mean']:+.4f} [{f_['lo']:+.4f},{f_['hi']:+.4f}] | {v['kind']} |")
    h5 = H["H5"]
    if "dF_pre" in h5:
        lines += ["", "## H5 — C vs OLLA (E5, secondary)", "",
                  f"state: **{h5['state']}** (pre {h5['leg_pre']}, tau {h5['leg_tau']}, "
                  f"transient {h5['leg_trans']}; censored C={h5['censored_C']}, D={h5['censored_D']})", "",
                  f"- dF_pre = {h5['dF_pre']['mean']:+.4f} [{h5['dF_pre']['lo']:+.4f},{h5['dF_pre']['hi']:+.4f}] (tol ±2 pp)",
                  f"- dTau = {h5['dTau']['mean']:+.1f} [{h5['dTau']['lo']:+.1f},{h5['dTau']['hi']:+.1f}] slots",
                  f"- dF_trans = {h5['dF_trans']['mean']:+.4f} [{h5['dF_trans']['lo']:+.4f},{h5['dF_trans']['hi']:+.4f}]"]
    else:
        lines += ["", "## H5 — NOT RUN", ""]

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    for name in ("H1_summary", "H2_summary", "H3", "H4", "H5"):
        v = H[name]
        print(name, v.get("state", ""),
              {kk: v[kk] for kk in ("supported", "contradicted", "inconclusive") if kk in v})
    print(f"\nreport: {out / 'report.md'}")


if __name__ == "__main__":
    main()
