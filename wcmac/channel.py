"""Channel, measurement-noise and decoding-stream generation.

Model (Design §4.2, §4.3; Protocol §3.1, §3.2):

    gamma_t = mu_t + rho (gamma_{t-1} - mu_{t-1}) + sigma_gamma sqrt(1-rho^2) z_t
    gamma_hat_t = gamma_t + n_t,     n_t ~ N(0, sigma_n^2)  i.i.d.
    u_t ~ U(0,1)                     decoding stream, keyed by (seed, t) only

The 200-slot burn-in is drawn and discarded at generation time (Protocol §3.1).
For E5/E9 the mean is time-varying: mu steps from 10 dB to 6 dB at t = 5000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .streams import draw_streams


@dataclass(frozen=True)
class Trace:
    seed: int
    true: np.ndarray       # (T,) true SNR, dB          -- simulator only
    noise: np.ndarray      # (T,) measurement noise, dB -- simulator only
    u: np.ndarray          # (T,) shared uniform draw   -- evaluation only
    scenario: str = ""
    split: str = ""

    @property
    def measured(self) -> np.ndarray:
        """gamma_hat_t = gamma_t + n_t.  Online-visible, from instant t+d."""
        return self.true + self.noise

    def __len__(self) -> int:
        return int(self.true.size)


def _mu_path(rho: float, mu_step: bool, n_total: int) -> np.ndarray:
    """mu_t for the (burn-in + evaluated) slots; step at evaluated slot 5000."""
    mu = np.full(n_total, C.MU)
    if mu_step:
        start = C.BURN_IN + C.T_STEP          # first evaluated slot with the new mean
        mu[start:] = 6.0
    return mu


def generate(seed: int, sigma_n: float, rho: float, mu_step: bool = False,
             scenario: str = "", split: str = "") -> Trace:
    ch, no, de = draw_streams(seed)

    n_total = C.BURN_IN + C.T_SLOTS
    mu = _mu_path(rho, mu_step, n_total)

    # stationary start, then the innovation sequence
    gamma0 = C.MU + C.SIGMA_GAMMA * ch.standard_normal()
    z = ch.standard_normal(n_total)

    gamma = np.empty(n_total)
    prev = gamma0
    prev_mu = C.MU
    scale = C.SIGMA_GAMMA * np.sqrt(1.0 - rho * rho)
    for t in range(n_total):
        prev = mu[t] + rho * (prev - prev_mu) + scale * z[t]
        prev_mu = mu[t]
        gamma[t] = prev

    true = gamma[C.BURN_IN:]
    if sigma_n > 0.0:
        noise = sigma_n * no.standard_normal(C.T_SLOTS)
    else:
        # draw nothing but keep the stream position identical across sigma_n so
        # that changing sigma_n cannot change any other stream (Protocol §4.2).
        no.standard_normal(C.T_SLOTS)
        noise = np.zeros(C.T_SLOTS)
    u = de.random(C.T_SLOTS)

    return Trace(seed=seed, true=true, noise=noise, u=u,
                 scenario=scenario, split=split)


def generate_train(seed: int, sigma_n: float, rho: float) -> Trace:
    return generate(seed, sigma_n, rho, scenario="train", split="train")
