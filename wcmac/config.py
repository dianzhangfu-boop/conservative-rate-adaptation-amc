"""Frozen configuration for EEEN3008J Topic 30 (Phase 3).

Every constant in this module is transcribed from the Phase 2 documents:

    Part A  Phase2_Research_Design.md
    Part B  Algorithm_Specification.md
    Part C  Experimental_Protocol.md

Nothing here may be tuned at run time.  The only run-time variation is the
`TraceSpec` / `Options` objects that are recorded verbatim in the run manifest.

Where a protocol instruction is under-specified, the implementation choice is
recorded in `DECISIONS` below and in `phase3/DEVIATIONS.md`; it is never
silently absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Physical model  (Design §4.2, §4.3; Protocol §3.1, §3.2)
# --------------------------------------------------------------------------

MU = 10.0            # dB, long-run mean SNR
SIGMA_GAMMA = 3.0    # dB, stationary std of the SNR process
BURN_IN = 200        # slots discarded at generation (Protocol §3.1)
T_SLOTS = 10_000     # evaluated slots per trace (Protocol §3.5)

MUHAT0 = 10.0        # dB, cold-start prediction gamma~_t for t <= d (DA-11)

# --------------------------------------------------------------------------
# Link and MCS  (Design §7.1; Protocol §3.3)
# --------------------------------------------------------------------------

N_MCS = 4
A_SLOPE = (1.0, 1.0, 1.0, 1.0)      # dB^-1
B_MID = (0.0, 4.0, 8.0, 12.0)       # dB
RATE = (0.5, 1.0, 2.0, 4.0)         # bit/s/Hz
BETA = 0.10                          # target BLER

# theta_m solves BLER_m(theta) = beta  =>  theta = b + ln((1-beta)/beta)/a
_LN_RATIO = math.log((1.0 - BETA) / BETA)          # ln 9 = 2.197224577...
THETA = tuple(b + _LN_RATIO / a for a, b in zip(A_SLOPE, B_MID))

# --------------------------------------------------------------------------
# Policy hyperparameters  (Design §8; Protocol §3.4)
# --------------------------------------------------------------------------

N_W = 100            # residual window length
ALPHA = 0.90         # quantile level
DELTA_MAX = 12.0     # dB, upper clip
N_MIN = 10           # minimum residuals before the quantile is used
DELTA_DOWN = 0.5     # dB, OLLA NACK step
DELTA_UP = DELTA_DOWN * BETA / (1.0 - BETA)   # dB, OLLA ACK step (~0.05556)

DELTA_GRID_STEP = 0.25
DELTA_GRID = tuple(i * DELTA_GRID_STEP for i in range(int(DELTA_MAX / DELTA_GRID_STEP) + 1))

# --------------------------------------------------------------------------
# Seeds  (Protocol §4.1)
# --------------------------------------------------------------------------

MASTER = 20_261_115
PRIME = 7919

SEED_OFFSET = {
    "train": 0,
    "main_test": 0,
    "main_val": 300_000,
    "stress_test": 900_000,
    "stress_val": 1_200_000,
    "e5": 1_500_000,
    "e8": 2_100_000,
    "e9": 2_400_000,
}
TRAIN_OFFSET = 1_800_000
E0_OFFSET = 2_700_000      # reserved diagnostic range: outside every analysis split


def e0_seed(i: int = 1) -> int:
    return MASTER + E0_OFFSET + i


def main_seed(k: int, i: int) -> int:
    """Main test seed: MASTER + 7919*k + i  (Protocol §4.1)."""
    return MASTER + PRIME * k + i


def main_val_seed(k: int, i: int) -> int:
    return MASTER + SEED_OFFSET["main_val"] + PRIME * k + i


def stress_seed(k: int, i: int) -> int:
    return MASTER + SEED_OFFSET["stress_test"] + PRIME * k + i


def stress_val_seed(k: int, i: int) -> int:
    return MASTER + SEED_OFFSET["stress_val"] + PRIME * k + i


def e5_seed(i: int) -> int:
    """E5 mu-step traces: i in 1..10 -> validation, i in 11..20 -> test."""
    return MASTER + SEED_OFFSET["e5"] + i


def train_seed(j: int) -> int:
    return MASTER + TRAIN_OFFSET + j


# --------------------------------------------------------------------------
# Scenarios  (Protocol §5.2, §5.2b)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    k: int
    sid: str
    sigma_n: float
    rho: float
    d: int
    split: str          # "main" | "stress"
    confirmatory: bool
    mu_step: bool = False
    note: str = ""

    @property
    def val_seed_fn(self):
        return main_val_seed if self.split == "main" else stress_val_seed

    @property
    def test_seed_fn(self):
        return main_seed if self.split == "main" else stress_seed


def build_scenarios() -> dict[int, Scenario]:
    """Enumerate the main matrix by formula (k = 6*i_sigma + 3*i_rho + i_d + 1)
    and append the five stress scenarios exactly as tabulated."""
    out: dict[int, Scenario] = {}
    sigmas = (0.0, 1.0, 2.0, 3.0)
    rhos = (0.95, 0.85)
    ds = (1, 3, 6)
    for i_s, s in enumerate(sigmas):
        for i_r, r in enumerate(rhos):
            for i_d, d in enumerate(ds):
                k = 6 * i_s + 3 * i_r + i_d + 1
                out[k] = Scenario(k, f"k{k:02d}", s, r, d, "main", True)

    stress = [
        (25, "S1", 1.0, 0.70, 3, True, False, "fast variation alone (H4)"),
        (26, "S2", 3.0, 0.70, 3, True, False, "fast variation + noise (H4)"),
        (27, "S3", 1.0, 0.85, 12, True, False, "long delay alone (H4)"),
        (28, "S4", 1.0, 0.70, 12, True, False, "both stressors (H4)"),
        (29, "S5", 5.0, 0.85, 3, False, False, "extreme noise (exploratory)"),
    ]
    for k, sid, s, r, d, conf, step, note in stress:
        out[k] = Scenario(k, sid, s, r, d, "stress", conf, step, note)
    return out


SCENARIOS = build_scenarios()

MAIN_KS = tuple(k for k in sorted(SCENARIOS) if SCENARIOS[k].split == "main")
STRESS_KS = tuple(k for k in sorted(SCENARIOS) if SCENARIOS[k].split == "stress")
CONFIRMATORY_KS = tuple(k for k in sorted(SCENARIOS) if SCENARIOS[k].confirmatory)

NOMINAL_K = 11          # sigma_n = 1, rho = 0.85, d = 3   (Protocol §5.2)

TRACES_PER_SCENARIO = 10
ESCALATED_TRACES_PER_SCENARIO = 20

# Training pool: (rho, sigma_n) -> number of traces, 8/7/8/7  (Design §6.4)
TRAINING_POOL = {(0.85, 1.0): 8, (0.85, 2.0): 7, (0.95, 1.0): 8, (0.95, 2.0): 7}
TRAIN_DS = (1, 3, 6, 12)

# E5 step location (Design §3 H5; Protocol §7.2)
T_STEP = 5000
PRE_STEP = (T_STEP - 500, T_STEP)
TRANSIENT = (T_STEP, T_STEP + 200)
POST_STEP = (T_STEP + 800, T_SLOTS)
RECOVERY_BAND = 0.02
RECOVERY_WINDOW = 100
RECOVERY_SUSTAIN = 100

# Pilot gate (Protocol §8.5; Design §12)
KAPPA = math.sqrt(9.0 / 3.325)       # = sqrt((n-1)/chi2_{0.05,n-1}), n = 10
GATE_FLOOR_DG = 0.025                # bit/s/Hz
GATE_FLOOR_DF = 0.005                # 0.5 pp, in fraction units
GATE_EFFECT_FRACTION = 0.20

# Multiplicity (Protocol §8.4)
HOLM_ALPHA = 0.05


@dataclass(frozen=True)
class Options:
    """Run-time options that do not alter the frozen design parameters.

    `bler` supplies an alternative MCS/BLER table for E7; `n_w` / `alpha` are
    overridden only by the E6 descriptive sweep.  Any other value here is a
    deviation and must be recorded.
    """

    n_w: int = N_W
    alpha: float = ALPHA
    delta_max: float = DELTA_MAX
    n_min: int = N_MIN
    a_slope: tuple = A_SLOPE
    b_mid: tuple = B_MID
    beta: float = BETA
    rates: tuple = RATE

    @property
    def theta(self):
        ln_ratio = math.log((1.0 - self.beta) / self.beta)
        return tuple(b + ln_ratio / a for a, b in zip(self.a_slope, self.b_mid))


DEFAULT_OPTIONS = Options()

# --------------------------------------------------------------------------
# Documented implementation decisions (also in phase3/DEVIATIONS.md)
# --------------------------------------------------------------------------

DECISIONS = {
    "D-P3-01": (
        "Pilot-gate H3 sd proxy. Protocol §8.5 step 1 says the H3 contrast uses "
        "the larger of the two per-cell sds, but Policy B-nominal is not run on "
        "validation traces (it is a test-only arm). The gate therefore uses the "
        "larger of the two per-cell sds of the C-vs-B-per-scenario paired "
        "differences at k=11 (sigma_n=1) and k=23 (sigma_n=3). No extra runs are "
        "added; the proxy is reported alongside the gate table."
    ),
    "D-P3-02": (
        "Pilot-gate trigger is read as a union over the four primary contrasts "
        "(dF,dG for C-vs-A and C-vs-B-per): a scenario escalates if ANY of them "
        "has projected half-width above its own threshold. This is the "
        "conservative reading of 'triggers if the projected half-width exceeds'."
    ),
    "D-P3-03": (
        "Policy A and Policy B(Delta=0) share one vectorised constant-margin code "
        "path; T5 still verifies the per-slot isomorphism empirically."
    ),
    "D-P3-04": (
        "E7's '10 BLER configs' are the four sensitivity families of Design §7.4 "
        "with their listed levels (3+3+2+2 = 10 configurations)."
    ),
    "D-P3-05": (
        "E5 uses Delta_init = Delta_fixed(k=11) for both C and D, because E5 runs "
        "at the nominal cell's parameters (Protocol §9). The untuned OLLA variant "
        "is primary and is labelled as such."
    ),
}
