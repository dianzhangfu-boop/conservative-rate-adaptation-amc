# -*- coding: utf-8 -*-
"""Report-only figure generation for the Phase 3 report.

Reads archived metrics; performs no experiment and no hypothesis decision.
Outputs PNGs under figures/ plus a checksum record.

Usage:  python make_figures.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artefact")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_rows(path):
    rows = []
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("G", "F_aggr", "BLER_obs"):
                r[k] = float(r[k])
            for k in ("k", "trace_index"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def mean_by_cell(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["k"], r["policy"])].append(r)
    out = {}
    for key, rs in g.items():
        out[key] = {
            "G": float(np.mean([x["G"] for x in rs])),
            "F": float(np.mean([x["F_aggr"] for x in rs])),
            "B": float(np.mean([x["BLER_obs"] for x in rs])),
        }
    return out


# ---------------------------------------------------------------- Figure 1
def figure_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fb", ec="#33507a"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                           linewidth=1.0, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.2, wrap=True)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-", color="#33507a"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=9, linewidth=1.0,
                                     linestyle=ls, color=color))

    box(0.15, 3.25, 1.35, 0.95, "Channel\n$\\gamma_t$")
    box(1.85, 3.25, 1.55, 0.95, "Measurement\n$\\hat{\\gamma}_t=\\gamma_t+n_t$")
    box(3.75, 3.25, 1.45, 0.95, "Delay $d$\n$\\hat{\\gamma}_{t-d}$")
    box(5.55, 3.25, 1.55, 0.95, "Predictor\n$\\tilde{\\gamma}_t=c_0+c_1\\hat{\\gamma}_{t-d}$")
    box(7.45, 3.25, 1.45, 0.95, "Margin\n$\\Delta_t=\\mathrm{clip}(Q_\\alpha(E_t))$")
    box(8.95, 3.25, 0.95, 0.95, "MCS\n$m_t$")

    arrow(1.50, 3.72, 1.85, 3.72)
    ax.text(1.67, 3.87, "$n_t$", ha="center", fontsize=7.5)
    arrow(3.40, 3.72, 3.75, 3.72)
    arrow(5.20, 3.72, 5.55, 3.72)
    arrow(7.10, 3.72, 7.45, 3.72)
    arrow(8.90, 3.72, 8.95, 3.72)

    # residual path
    box(5.55, 1.65, 1.55, 0.85, "Residual\n$e_s=\\tilde{\\gamma}_s-\\hat{\\gamma}_s$", fc="#f4f9ee", ec="#4d6b2f")
    arrow(6.33, 3.25, 6.33, 2.50)
    ax.text(6.45, 2.88, "$s\\leq t-d$", fontsize=7.5)
    arrow(7.10, 2.08, 7.95, 3.25)

    # evaluation path
    box(1.85, 0.35, 2.0, 0.85, "Link evaluation\n$\\mathrm{BLER}_{m_t}(\\gamma_t)$", fc="#fff4e6", ec="#9a5b13")
    box(4.25, 0.35, 2.55, 0.85, "Metrics\n$G$, $\\mathrm{BLER}_{obs}$, $F_{aggr}$", fc="#fff4e6", ec="#9a5b13")
    # selected MCS feeds the link evaluation (route below the residual box)
    arrow(9.40, 3.25, 9.40, 1.45)
    arrow(9.40, 1.45, 3.95, 1.45)
    arrow(3.95, 1.45, 3.85, 1.10)
    ax.text(6.55, 1.52, "$m_t$", fontsize=7.5)
    arrow(3.85, 0.78, 4.25, 0.78)
    arrow(0.82, 3.25, 0.82, 0.78)
    arrow(0.82, 0.78, 1.85, 0.78)

    ax.annotate("$\\gamma_t$ is simulator-only; the controller never reads it",
                xy=(0.82, 3.25), xytext=(0.15, 4.55),
                fontsize=7.5, color="#9a1c1c",
                arrowprops=dict(arrowstyle="-", color="#9a1c1c", lw=0.8, ls="--"))
    ax.text(5.0, 4.72, "Causal decision path (solid) and offline evaluation boxes (bottom)",
            ha="center", fontsize=8.5)

    fig.savefig(os.path.join(FIG, "fig1_pipeline.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def figure_tradeoff(rows):
    m = mean_by_cell(rows)
    ks = sorted({k for k, _ in m})
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    styles = {
        "A": dict(marker="^", color="#7a7a7a", label="A: prediction-only"),
        "B_per": dict(marker="s", color="#2b6cb0", label="B: fixed margin (per scenario)"),
        "C": dict(marker="o", color="#c53030", label="C: adaptive residual quantile"),
    }
    for pol, st in styles.items():
        xs = [m[(k, pol)]["F"] for k in ks if (k, pol) in m]
        ys = [m[(k, pol)]["G"] for k in ks if (k, pol) in m]
        ax.scatter(xs, ys, s=28, alpha=0.85, edgecolor="white", linewidth=0.4, **st)
    # light arrows A -> C per scenario
    for k in ks:
        if (k, "A") in m and (k, "C") in m:
            ax.annotate("", xy=(m[(k, "C")]["F"], m[(k, "C")]["G"]),
                        xytext=(m[(k, "A")]["F"], m[(k, "A")]["G"]),
                        arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6, alpha=0.7))
    ax.set_xlabel(r"Aggressive-selection fraction $F_{\mathrm{aggr}}$")
    ax.set_ylabel("Model-expected goodput $G$ (bit/s/Hz)")
    ax.set_title("Reliability–goodput trade-off across the 24 main cells")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig2_tradeoff.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def figure_h3():
    H = json.load(io.open(os.path.join(ART, "analysis", "hypotheses.json"), encoding="utf-8"))
    pts = [("1.0", 1.0), ("3.0", 3.0)]
    means, los, his = [], [], []
    for key, _ in pts:
        d = H["H3"]["D_by_sigma"][key]
        means.append(d["mean"]); los.append(d["lo"]); his.append(d["hi"])
    x = np.array([p[1] for p in pts])
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(los),
                                np.array(his) - np.array(means)],
                fmt="o", color="#c53030", ecolor="#c53030", capsize=4,
                markersize=5, lw=1.2)
    ax.axhline(0, color="#444444", lw=0.8, ls="--")
    ax.set_xlim(0.4, 3.6)
    ax.set_xticks([1, 3])
    ax.set_xlabel(r"Measurement-noise standard deviation $\sigma_n$ (dB)")
    ax.set_ylabel(r"$D(\sigma_n)=F_C-F_{B\mathrm{nom}}$")
    ax.set_title("H3: noise-dependent relative reliability (95 % CI)")
    ax.grid(alpha=0.25)
    c = H["H3"]["contrast_D3_minus_D1"]
    ax.text(0.03, 0.05,
            "D(3) − D(1) = %.4f [%.4f, %.4f]" % (c["mean"], c["lo"], c["hi"]),
            transform=ax.transAxes, fontsize=8)
    fig.savefig(os.path.join(FIG, "fig3_h3_noise.png"))
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def figure_h4():
    rows = []
    with io.open(os.path.join(ART, "analysis", "h4_contrasts.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "G":
                rows.append(r)
    labels = {"25": "S1", "26": "S2", "27": "S3", "28": "S4"}
    rows.sort(key=lambda r: int(r["k"]))
    means = np.array([float(r["mean"]) for r in rows])
    los = np.array([float(r["ci_lo"]) for r in rows])
    his = np.array([float(r["ci_hi"]) for r in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.errorbar(x, means, yerr=[means - los, his - means], fmt="o",
                color="#2b6cb0", ecolor="#2b6cb0", capsize=4, markersize=5, lw=1.2)
    ax.axhline(0.05, color="#c53030", lw=0.9, ls="--")
    ax.text(0.02, 0.044, "practical threshold 0.05", color="#c53030",
            fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#444444", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[r["k"]] for r in rows])
    ax.set_ylabel(r"$\Delta G=G_C-G_{B}$ (bit/s/Hz)")
    ax.set_title("H4: stress-cell goodput differences (95 % CI)")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(os.path.join(FIG, "fig4_h4_stress.png"))
    plt.close(fig)


def main():
    rows = load_rows(os.path.join(ART, "test", "metrics_per_run.csv"))
    figure_pipeline()
    figure_tradeoff(rows)
    figure_h3()
    figure_h4()
    checks = {}
    for name in ("fig1_pipeline.png", "fig2_tradeoff.png", "fig3_h3_noise.png", "fig4_h4_stress.png"):
        p = os.path.join(FIG, name)
        checks[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    with io.open(os.path.join(FIG, "checksums.json"), "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, sort_keys=True)
    print("figures written to", FIG)
    for k, v in checks.items():
        print(" ", k, v[:16])


if __name__ == "__main__":
    main()
