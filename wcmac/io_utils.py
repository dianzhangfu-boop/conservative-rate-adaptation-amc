"""Artefact I/O: checksums, manifests, atomic writes, code versioning."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # phase3/
ART = ROOT / "artefact"


def ensure(*parts: str) -> Path:
    p = ART.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> str:
    """SHA-256 over every .py file under wcmac/ and scripts/, path-sorted."""
    h = hashlib.sha256()
    files = sorted(list((ROOT / "wcmac").glob("*.py")) + list((ROOT / "scripts").glob("*.py")))
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_default)
    os.replace(tmp, path)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def append_jsonl(path: str | Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=_default) + "\n")


def write_csv(path: str | Path, rows: list[dict], columns: list[str] | None = None) -> None:
    import csv
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if columns:
        cols = columns
    else:                       # union of keys, first-seen order: rows differ
        seen, cols = set(), []  # (e.g. experiment-specific optional fields)
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})


def _fmt(v):
    if isinstance(v, float):
        return repr(v)
    return v


def env_record() -> dict:
    import numpy
    import scipy
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def env_txt() -> None:
    rec = env_record()
    ensure()
    (ART / "env.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in rec.items()) + "\n", encoding="utf-8"
    )
