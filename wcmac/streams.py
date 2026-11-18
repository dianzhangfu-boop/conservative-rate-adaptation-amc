"""Deterministic RNG stream derivation (Protocol §4.2, §4.3).

Three independent streams per trace seed are derived by hashing the seed with a
fixed label.  hashlib is used rather than Python's hash() because hash() is
salted per process and would break reproducibility (T11).
"""

from __future__ import annotations

import hashlib

import numpy as np

LABELS = ("channel", "noise", "decode")


def _label_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "big")


def stream(seed: int, label: str) -> np.random.Generator:
    """Return the Generator for (seed, label). Deterministic across processes.

    PCG64 + SeedSequence is used because its bit stream is stable across NumPy
    versions for a fixed seed sequence (unlike legacy RandomState APIs).
    """
    if label not in LABELS:
        raise ValueError(f"unknown stream label {label!r}")
    ss = np.random.SeedSequence([int(seed), _label_int(label)])
    return np.random.Generator(np.random.PCG64(ss))


def draw_streams(seed: int):
    return tuple(stream(seed, lab) for lab in LABELS)


def checksum_streams(seed: int) -> dict:
    """Checksums used by T5 to prove the streams do not depend on the policy."""
    ch, no, de = draw_streams(seed)
    return {
        "channel_first8": ch.standard_normal(8).tolist(),
        "noise_first8": no.standard_normal(8).tolist(),
        "decode_first8": de.random(8).tolist(),
    }
