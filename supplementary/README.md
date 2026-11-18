# Supplementary artefacts

These files are byte-for-byte copies of the frozen result files, renamed for
readability. The originals remain in the project `phase3/artefact/` tree and
were not modified.

| File | Original | Content |
|---|---|---|
| `E0_TEST_REPORT.md` | `artefact/e0/test_report.md` | E0 gate, T1–T13, 13/13 PASS |
| `PILOT_GATE.md` | `artefact/gate/pilot_gate.md` | Pilot-gate table and decision |
| `PILOT_GATE.json` | `artefact/gate/pilot_gate.json` | Machine-readable gate evidence |
| `VALIDATION_TUNING.json` | `artefact/validation/tuning.json` | Per-scenario Δ_fixed and F* |
| `VALIDATION_RUNLOG.json` | `artefact/validation_runlog.json` | Validation run count and hashes |
| `PREDICTOR.json` | `artefact/predictor.json` | Frozen predictor coefficients and fit ledger |
| `HYPOTHESES.json` | `artefact/analysis/hypotheses.json` | H1–H5 classifications and CIs |
| `H1_CONTRASTS.csv` | `artefact/analysis/h1_contrasts.csv` | H1 per-cell contrasts and MDE |
| `H2_CONTRASTS.csv` | `artefact/analysis/h2_contrasts.csv` | H2 leg-level contrasts |
| `H4_CONTRASTS.csv` | `artefact/analysis/h4_contrasts.csv` | H4 stress-cell contrasts |
| `ANALYSIS_REPORT.md` | `artefact/analysis/report.md` | Generated analysis summary |
| `TEST_METRICS_PER_RUN.csv` | `artefact/test/metrics_per_run.csv` | Per-run metrics (analysis input) |
| `TEST_RUNLOG.json` | `artefact/test/runlog.json` | Test-run counts and archive hashes |
| `DESCRIPTIVE_RUNS.csv` | `artefact/descriptive/descriptive_runs.csv` | E6/E7 descriptive runs |
| `FROZEN.json` | `artefact/FROZEN.json` | Freeze manifest and checksums |
| `ENV.txt` | `artefact/env.txt` | Build/run environment |

## Excluded: test trace archive

`artefact/test/test_traces.npz` is 61 MB and is not copied into this package.
Its SHA-256 is recorded in `TEST_RUNLOG.json` as `traces_npz_sha256`; the file
remains in the project at `phase3/artefact/test/test_traces.npz`. Excluding it
keeps the package small and does not alter the archive.

## Interpretation

- `TEST_METRICS_PER_RUN.csv` is the input to all reported H1–H5 statistics; one
  row is one policy on one trace.
- `HYPOTHESES.json` contains the final classifications and confidence intervals.
- The frozen code hash and artefact checksums are in `FROZEN.json`; the report
  and checklist record the verification result.
- No file here was recompressed, edited or regenerated for Phase 6.
