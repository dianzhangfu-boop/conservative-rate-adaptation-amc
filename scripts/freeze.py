#!/usr/bin/env python
"""Freeze every decision-relevant artefact before the test set is opened.

Writes artefact/FROZEN.json containing SHA-256 checksums of the predictor, the
per-scenario tuning, the pilot-gate decision, the configuration module and the
code, together with the full parameter record.  scripts/run_test.py refuses to
run unless FROZEN.json exists and the current code version matches the frozen
one, which is what makes post-hoc tuning of the test set structurally
impossible.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wcmac import config as C
from wcmac.io_utils import (ART, code_version, ensure, env_record, read_json,
                            sha256_file, write_json)

REQUIRED = {
    "predictor": "predictor.json",
    "tuning": "validation/tuning.json",
    "gate": "gate/pilot_gate.json",
    "validation_metrics": "validation/metrics_per_run.csv",
}


def main():
    missing = [str(ART / p) for p in REQUIRED.values() if not (ART / p).exists()]
    if missing:
        raise SystemExit("cannot freeze, missing artefacts:\n  " + "\n  ".join(missing))

    predictor = read_json(ART / "predictor.json")
    tuning = read_json(ART / "validation" / "tuning.json")
    gate = read_json(ART / "gate" / "pilot_gate.json")

    manifest = {
        "frozen_at": env_record()["timestamp"],
        "code_version": code_version(),
        "environment": env_record(),
        "checksums": {name: sha256_file(ART / rel) for name, rel in REQUIRED.items()},
        "predictors": predictor["predictors"],
        "hyperparameters": {
            "N_w": C.N_W, "alpha": C.ALPHA, "delta_max": C.DELTA_MAX,
            "n_min": C.N_MIN, "beta": C.BETA,
            "delta_down": C.DELTA_DOWN, "delta_up": C.DELTA_UP,
            "mu": C.MU, "sigma_gamma": C.SIGMA_GAMMA, "T": C.T_SLOTS,
            "burn_in": C.BURN_IN, "muh0": C.MUHAT0,
            "theta": list(C.THETA), "rates": list(C.RATE),
            "kappa": C.KAPPA,
            "gate_floors": {"dG": C.GATE_FLOOR_DG, "dF": C.GATE_FLOOR_DF,
                            "effect_fraction": C.GATE_EFFECT_FRACTION},
        },
        "delta_fixed": {k: v["delta_fixed"] for k, v in tuning.items()},
        "delta_init": {k: v["delta_fixed"] for k, v in tuning.items()},
        "scenarios": {
            str(k): {"sid": s.sid, "sigma_n": s.sigma_n, "rho": s.rho, "d": s.d,
                     "split": s.split, "confirmatory": s.confirmatory,
                     "mu_step": s.mu_step}
            for k, s in C.SCENARIOS.items()
        },
        "seeds": {
            "master": C.MASTER,
            "main_test": {str(k): [C.main_seed(k, i) for i in range(1, 11)]
                          for k in C.MAIN_KS},
            "stress_test": {str(k): [C.stress_seed(k, i) for i in range(1, 11)]
                            for k in C.STRESS_KS},
            "e5_test": [C.e5_seed(i) for i in range(11, 21)],
        },
        "escalation": {
            "escalated_scenarios": gate["escalated"],
            "extra_runs": gate["extra_test_traces_runs"],
            "rule": "Protocol 8.5; see artefact/gate/pilot_gate.md",
        },
        "metric_definitions": {
            "G": "(1/T) sum R_{m_t} (1 - BLER_{m_t}(gamma_t))",
            "G_real": "(1/T) sum R_{m_t} B_t",
            "BLER_obs": "1 - (1/T) sum B_t",
            "F_aggr": "(1/T) sum 1{BLER_{m_t}(gamma_t) > beta}",
        },
    }
    write_json(ART / "FROZEN.json", manifest)
    print(f"frozen: {ART / 'FROZEN.json'}")
    print(f"code_version: {manifest['code_version']}")
    print(f"escalated: {gate['escalated'] or 'none'}")


if __name__ == "__main__":
    main()
