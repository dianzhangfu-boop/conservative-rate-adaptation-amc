# Conservative Rate Adaptation with Uncertain Channel Estimates

Code and compact reproducibility artefacts for the EEEN3008J Wireless
Communications group project (2026/27), Topic 30.

The project evaluates four adaptive modulation and coding policies under noisy
channel measurements and delayed feedback:

- A: prediction-only AMC;
- B: validation-tuned fixed SNR margin;
- C: prediction-error-driven adaptive residual-quantile margin;
- D: outer-loop link adaptation (OLLA).

## Repository contents

- `wcmac/`: channel, predictor, MCS, policy and statistics implementation;
- `scripts/`: E0, validation, pilot-gate, freeze, test and analysis drivers;
- `supplementary/`: frozen configuration and compact archived result tables;
- `figures/`: report figures and their checksum manifest;
- `make_figures.py`: regenerates the figures from archived metrics.

The 61 MB raw test-trace archive is intentionally not included. The compact
per-run metrics and analysis outputs needed to inspect the reported results are
included under `supplementary/`.

## Environment

The experimental environment used Python 3.11.9, NumPy 1.26.4 and SciPy
1.17.1. Matplotlib 3.11.1 is needed only to regenerate the figures.

```powershell
python -m pip install -r requirements.txt
```

## Execution order

The original full project layout used the following order:

```powershell
python -I scripts/run_e0.py
python -I scripts/run_validation.py
python -I scripts/run_pilot_gate.py
python -I scripts/freeze.py
python -I scripts/run_test.py
python -I scripts/run_descriptive.py
python -I scripts/analyse.py
```

The full simulation requires the original `artefact/` tree and raw trace
archive, which are not published here. Figures can be regenerated from the
included compact artefacts after restoring the original report-package layout.

## Integrity and limitations

The frozen experimental code hash is
`3b6ba8ef7c152e91eb9c0f66117a4ab5163ea9832d830fc080643e5ddaf5e115`.
It covers `wcmac/*.py` and `scripts/*.py` and matches
`supplementary/FROZEN.json`.

The project report discloses protocol deviation D-P3-15: the complete test set
was accessed before the final freeze. The code and archived outputs are
published for transparency; the repository does not remove that limitation or
turn the study into a standard-compliant or deployment validation.

No credentials, API tokens, raw personal data or unrelated files are included.
