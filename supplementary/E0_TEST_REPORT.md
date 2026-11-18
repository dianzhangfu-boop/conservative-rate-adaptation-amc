# E0 validity gate — test report

## T1 — true-SNR leakage sentinel: **PASS**

Decisions of A, B and C are bit-identical under garbage true SNR; D is bit-identical under outcome replay; the poison demonstrably changes outcomes.

```json
{
  "A": {
    "mcs_bit_identical": true,
    "delta_bit_identical": true
  },
  "B": {
    "mcs_bit_identical": true,
    "delta_bit_identical": true
  },
  "C": {
    "mcs_bit_identical": true,
    "delta_bit_identical": true
  },
  "D": {
    "mcs_bit_identical_under_outcome_replay": true,
    "delta_bit_identical_under_outcome_replay": true,
    "note": "OLLA consumes outcomes by design; replay isolates the channel"
  },
  "poison_reached_evaluator": true
}
```

## T2 — future-feedback leakage: **PASS**

No read beyond t-d in 19991 feed accesses; a permissive feed that would return garbage for future indices changes nothing; the guard raises on a direct future read.

```json
{
  "n_reads": 19991,
  "violations": 0,
  "decisions_unchanged_under_permissive_feed": true,
  "guard_raises": true
}
```

## T3 — feedback-delay alignment: **PASS**

Timing table reproduced exactly for d=3 and d=6 on one realisation; a predictor fitted for another d is rejected; the decision logs differ only through availability.

```json
{
  "expected": {
    "d=3": {
      "first": 7,
      "handover": 16,
      "full": 106
    },
    "d=6": {
      "first": 13,
      "handover": 22,
      "full": 112
    }
  },
  "observed": {
    "d=3": {
      "first": 7,
      "handover": 16,
      "full": 106
    },
    "d=6": {
      "first": 13,
      "handover": 22,
      "full": 112
    }
  },
  "predictor_mismatch_raises": true,
  "decisions_differ": true,
  "bank": {
    "1": {
      "c0": 2.9180333882807847,
      "c1": 0.7077100057194222,
      "d": 1,
      "n_pairs": 299970
    },
    "3": {
      "c0": 4.226453613201395,
      "c1": 0.5765943723441688,
      "d": 3,
      "n_pairs": 299910
    },
    "6": {
      "c0": 5.67249415633368,
      "c1": 0.4316893005146245,
      "d": 6,
      "n_pairs": 299820
    },
    "12": {
      "c0": 7.3737678979189605,
      "c1": 0.26116600262392997,
      "d": 12,
      "n_pairs": 299640
    }
  }
}
```

## T4 — OLLA zero-drift sanity: **PASS**

Directions match the spec; the clipped single-path drift is within 1e-4 dB/slot; an unclipped 20-seed test locates the mean increment at 0 and excludes the wrong-ratio value -0.4 dB/slot.

```json
{
  "direction_ok": true,
  "literal_drift_dB_per_slot": 5.555555555579871e-07,
  "unclipped_mean_increment": -0.0001755555555555676,
  "unclipped_ci95": [
    -0.00045403051622040715,
    0.00010291940510927198
  ],
  "wrong_ratio_increment": -0.4,
  "ci_excludes_wrong_ratio": true,
  "note": "the literal tolerance is uninformative when the clip is active (any bounded path satisfies it), hence check (ii)"
}
```

## T5 — randomness fairness and A == B(0): **PASS**

Policy runs do not consume or mutate the gamma/n/u streams; A and B(Delta=0) are slot-by-slot identical; the decode stream does not move when sigma_n changes.

```json
{
  "stream_checksums": {
    "gamma": "d3a13ee4b75e9e61",
    "noise": "2907464ba0892d69",
    "u": "c16996c8e57ed4d4"
  },
  "streams_unchanged": true,
  "A_equals_B0_all_slots": true,
  "u_independent_of_sigma_n": true,
  "noise_depends_on_sigma_n": true,
  "B3_differs_from_B0": true
}
```

## T6 — dataset split integrity: **PASS**

All seven seed sets are pairwise disjoint and the fit ledger contains training trace IDs only. The archive-checksum sub-check is performed at test time, when test traces exist.

```json
{
  "sizes": {
    "main_test": 240,
    "main_val": 480,
    "stress_test": 50,
    "stress_val": 100,
    "e5": 20,
    "train": 30,
    "e0": 1
  },
  "overlaps": {},
  "fit_ledger_training_only": true,
  "deferred": "archive checksum of test traces (T6c) runs after the test traces are archived"
}
```

## T7 — residual identity at zero noise: **PASS**

With sigma_n = 0 the observable residual equals the true prediction error to machine precision for every slot s >= d+1, and no residual exists for s <= d.

```json
{
  "max_abs_error": 0.0,
  "no_residual_before_d_plus_1": true,
  "predictor": {
    "c0": 0.2595345921543757,
    "c1": 0.974080505170317,
    "d": 3,
    "n_pairs": 9997
  }
}
```

## T8 — quantile convention correctness: **PASS**

Type-7 interpolation matches an independent hand reference for m = 10 and agrees with the analytic Gaussian quantile on 1e6 draws.

```json
{
  "m10_reference": {
    "0.5": 0.43000000000000005,
    "0.9": 2.2359999999999993,
    "0.99": 3.7426000000000004
  },
  "m10_observed": {
    "0.5": 0.43000000000000005,
    "0.9": 2.2359999999999993,
    "0.99": 3.7426000000000004
  },
  "gaussian_abs_error": {
    "0.5": 0.0001291238822632118,
    "0.9": 3.864917252971445e-05,
    "0.99": 0.0015668170812763726
  }
}
```

## T9 — BLER monotonicity and threshold identity: **PASS**

Each BLER curve is strictly decreasing on the 0.01 dB grid and theta_m solves BLER_m(theta) = beta to better than 1e-9.

```json
{
  "strictly_decreasing": {
    "1": true,
    "2": true,
    "3": true,
    "4": true
  },
  "theta_abs_error": {
    "1": 2.7755575615628914e-17,
    "2": 5.551115123125783e-17,
    "3": 5.551115123125783e-17,
    "4": 5.551115123125783e-17
  }
}
```

## T10 — warm-up and window-fill timing: **PASS**

First residual at 2d+1, margin handover at 2d+n_min and window full at 2d+N_w for every d in {1,3,6,12}.

```json
{
  "rows": [
    {
      "d": 1,
      "expected": [
        3,
        12,
        102
      ],
      "observed": [
        3,
        12,
        102
      ],
      "match": true
    },
    {
      "d": 3,
      "expected": [
        7,
        16,
        106
      ],
      "observed": [
        7,
        16,
        106
      ],
      "match": true
    },
    {
      "d": 6,
      "expected": [
        13,
        22,
        112
      ],
      "observed": [
        13,
        22,
        112
      ],
      "match": true
    },
    {
      "d": 12,
      "expected": [
        25,
        34,
        124
      ],
      "observed": [
        25,
        34,
        124
      ],
      "match": true
    }
  ]
}
```

## T11 — determinism and reproducibility: **PASS**

Same seed reproduces the trace and every metric exactly; a different seed produces a different trace.

```json
{
  "same_seed_metrics_identical": true,
  "same_seed_trace_identical": true,
  "different_seed_trace_differs": true
}
```

## T12 — clip and forced-lowest accounting: **PASS**

Delta stays in [0, 12]; the three clip classes partition the trace; forced-lowest slots are counted and stay in the F_aggr denominator.

```json
{
  "clip_hi_frac": 0.0,
  "clip_lo_frac": 0.0,
  "interior_frac": 1.0,
  "bounds_ok": true,
  "forced_lowest_slots": 108,
  "forced_lowest_frac": 0.0108,
  "denominator_is_T": true
}
```

## T13 — cold-start behaviour: **PASS**

Slots 1..d use gamma_tilde = 10 dB for every policy with the policy's initial margin; the pending buffer forms no residual before instant 2d+1.

```json
{
  "policies": {
    "A": {
      "gt_ok": true,
      "delta_ok": true
    },
    "B": {
      "gt_ok": true,
      "delta_ok": true
    },
    "C": {
      "gt_ok": true,
      "delta_ok": true
    },
    "D": {
      "gt_ok": true,
      "delta_ok": true
    }
  },
  "first_residual_instant": 7,
  "expected_first_residual": 7
}
```
