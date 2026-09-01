# TAOT-SRC v4 frozen results

The protocol-corrected v4 rerun has been completed and audited. The files in `results/rerun_full/` and `results/derived_rerun/` are the authoritative numerical record for this repository version.

## Selected real-data hyperparameters

Validation-only selection on Digits seeds `0--4` gives:

```text
lambda  = 0.0
epsilon = 0.01
eta     = 0.5
```

## Held-out rotation results

Mean +/- standard deviation over held-out seeds `5--19`:

| Dataset | Rotation | SRC | OT-1NN | RBF-SVM | Linear-TAOT | TAOT-SRC |
|---|---:|---:|---:|---:|---:|---:|
| Digits | clean | 67.53 +/- 3.46 | 72.20 +/- 5.05 | 71.27 +/- 4.04 | 71.73 +/- 5.24 | **76.00 +/- 5.25** |
| Digits | +/-10 deg | 44.60 +/- 5.07 | 44.00 +/- 6.70 | 44.73 +/- 5.62 | 29.53 +/- 3.72 | **55.73 +/- 5.36** |
| Digits | +/-20 deg | 40.47 +/- 5.44 | 38.47 +/- 5.99 | 39.80 +/- 4.28 | 26.67 +/- 4.42 | **48.47 +/- 4.96** |
| LFW-Face | clean | 84.22 +/- 6.33 | 80.78 +/- 4.54 | 86.89 +/- 5.38 | 88.11 +/- 5.73 | **89.89 +/- 4.06** |
| LFW-Face | +/-10 deg | 68.00 +/- 9.78 | 76.22 +/- 5.78 | **84.00 +/- 10.32** | 76.44 +/- 11.30 | 82.00 +/- 10.53 |
| LFW-Face | +/-20 deg | 69.00 +/- 7.96 | 69.11 +/- 5.36 | **80.44 +/- 6.81** | 77.00 +/- 8.96 | 79.22 +/- 6.17 |

## Paired TAOT-SRC versus signed SRC

| Dataset | Condition | Delta (pp) | Paired two-sided t-test p |
|---|---:|---:|---:|
| Digits | clean | +8.47 | 2.65e-6 |
| Digits | +/-10 deg | +11.13 | 2.31e-7 |
| Digits | +/-20 deg | +8.00 | 3.86e-5 |
| LFW-Face | clean | +5.67 | 2.85e-4 |
| LFW-Face | +/-10 deg | +14.00 | 3.59e-5 |
| LFW-Face | +/-20 deg | +10.22 | 8.84e-5 |

## Few-shot TAOT-SRC results

| Dataset | Shots | TAOT-SRC | Effective atoms |
|---|---:|---:|---:|
| Digits | 1 | 52.87 +/- 6.33 | 1.00 +/- 0.00 |
| Digits | 2 | 65.07 +/- 6.03 | 1.64 +/- 0.03 |
| Digits | 3 | 70.60 +/- 6.33 | 2.05 +/- 0.07 |
| Digits | 5 | 76.00 +/- 5.25 | 2.60 +/- 0.08 |
| Digits | 10 | 82.73 +/- 3.73 | 3.30 +/- 0.13 |
| LFW-Face | 1 | 70.11 +/- 14.34 | 1.00 +/- 0.00 |
| LFW-Face | 2 | 83.67 +/- 6.99 | 1.46 +/- 0.10 |
| LFW-Face | 3 | 86.00 +/- 6.69 | 1.86 +/- 0.12 |
| LFW-Face | 5 | 89.89 +/- 4.06 | 2.34 +/- 0.13 |
| LFW-Face | 10 | 90.78 +/- 4.08 | 3.11 +/- 0.13 |

## Synthetic mechanism test

```text
signed SRC                 72.10 +/- 3.37 %
nonnegative SRC            72.67 +/- 3.43 %
wrong-geometry OT          68.83 +/- 6.87 %
intrinsic-geodesic TAOT    77.20 +/- 3.51 %
paired p vs signed SRC     2.00e-5
```

For the complete per-seed outputs, summaries, and statistical tests, see `results/rerun_full/` and `results/derived_rerun/`.
