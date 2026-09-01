# TAOT-SRC

**Topology-Aware Optimal Transport for Sparse Representation Classification**

This repository contains the protocol-corrected **v4** implementation and frozen reproducibility record for TAOT-SRC. The method replaces Euclidean class reconstruction in sparse representation classification with class-specific mixtures optimized using the transport cost induced by an entropically regularized optimal-transport coupling whose ground cost respects the topology of the feature support.

For unsigned HOG descriptors, orientation is periodic on `[0, pi)`. TAOT-SRC therefore uses a circular geodesic angular cost so that the first and last orientation bins are treated as neighbors rather than as distant endpoints.

## Reproducibility status

The v4 protocol corrections have been applied, validation has been rerun, held-out experiments have been regenerated, and the results have been audited against the manuscript. The authoritative outputs are stored in:

```text
results/validation_rerun/
results/rerun_full/
results/derived_rerun/
figures_rerun/
```

## Data policy

The public GitHub repository **does not redistribute the original image datasets or the derived `data/features/` cache files**.

The real-data experiments use datasets obtained through maintained Python interfaces:

- **Optical Recognition of Handwritten Digits** via `sklearn.datasets.load_digits()`.
- **LFW face/non-face subset** via `skimage.data.lfw_subset()` (the 200-image subset used in the paper: 100 face and 100 non-face images).

The repository does ship the deterministic v4 manifests under `data/manifests/`. These files specify the exact train/test indices and paired rotation angles used by the experiments.

No manual dataset download is required. The derived HOG and synthetic caches are created locally with:

```bash
python code/prepare_reference_data.py --root .
```

or automatically on the first:

```bash
python verify.py
```

The generated caches are written to `data/features/`, which is ignored by Git.

## What changed in v4

1. A seed defines one class-wise split shared by clean, `+/-10 deg`, and `+/-20 deg` evaluation.
2. Rotation randomness is independent of split randomness.
3. Each test sample receives one factor `u ~ Uniform[-1,1]`; its perturbations are `10u` and `20u` degrees, so rotation severity is paired sample-by-sample.
4. Few-shot experiments use a fixed test set for 1/2/3/5/10 shots, with nested training subsets drawn from one 10-shot pool.
5. Coefficient entropy is implemented exactly as `H(w) = -sum_j w_j log(w_j + 1e-9)`.
6. Real-data OT methods share `eta=0.5` and `epsilon=0.01`; pairwise OT baselines use 60 Sinkhorn iterations, while TAOT uses 30 during optimization and 50 for final scoring.
7. Ground-cost normalization uses a `1e-12` denominator safeguard.

See [`PROTOCOL_CORRECTION_V4.md`](PROTOCOL_CORRECTION_V4.md) and [`METHOD_DEFINITION.md`](METHOD_DEFINITION.md) for the exact protocol and method definitions.

## Authoritative v4 configuration

Validation uses only Digits seeds `0--4`. The final real-data configuration is:

```text
lambda  = 0.0
epsilon = 0.01
eta     = 0.5
```

Held-out real-data experiments use seeds `5--19`. No held-out result is used for hyperparameter selection.

The synthetic mechanism experiment separately uses:

```text
epsilon = 0.03
lambda  = 2.5e-4
```

## Main results

Held-out accuracy is mean +/- standard deviation over 15 seeds.

| Dataset | Condition | SRC | TAOT-SRC |
|---|---:|---:|---:|
| Digits | clean | 67.53 +/- 3.46 | **76.00 +/- 5.25** |
| Digits | +/-10 deg | 44.60 +/- 5.07 | **55.73 +/- 5.36** |
| Digits | +/-20 deg | 40.47 +/- 5.44 | **48.47 +/- 4.96** |
| LFW-Face | clean | 84.22 +/- 6.33 | **89.89 +/- 4.06** |
| LFW-Face | +/-10 deg | 68.00 +/- 9.78 | **82.00 +/- 10.53** |
| LFW-Face | +/-20 deg | 69.00 +/- 7.96 | **79.22 +/- 6.17** |

Paired two-sided t-tests comparing TAOT-SRC with canonical signed SRC give:

| Dataset | Condition | Improvement (pp) | p-value |
|---|---:|---:|---:|
| Digits | clean | +8.47 | 2.65e-6 |
| Digits | +/-10 deg | +11.13 | 2.31e-7 |
| Digits | +/-20 deg | +8.00 | 3.86e-5 |
| LFW-Face | clean | +5.67 | 2.85e-4 |
| LFW-Face | +/-10 deg | +14.00 | 3.59e-5 |
| LFW-Face | +/-20 deg | +10.22 | 8.84e-5 |

The synthetic mechanism test gives:

```text
signed SRC                 72.10 +/- 3.37 %
nonnegative SRC            72.67 +/- 3.43 %
wrong-geometry OT          68.83 +/- 6.87 %
intrinsic-geodesic TAOT    77.20 +/- 3.51 %
paired p vs signed SRC     2.00e-5
```

Detailed tables are available in [`RESULTS.md`](RESULTS.md) and under `results/derived_rerun/`.

## Environment

The audited reference environment is:

```text
Python          3.13.5
NumPy           2.3.5
pandas          2.2.3
SciPy           1.17.0
scikit-learn    1.8.0
scikit-image    0.26.0
PyTorch         2.10.0 CPU
pytest          9.0.2
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The experiments were also validated in the Conda environment used during development:

```bash
conda activate taot_exact
pip install -r requirements.txt
```

## Quick verification

Because `data/features/` is intentionally absent from GitHub, the first verification automatically reconstructs the required feature caches from the frozen manifests and public dataset interfaces:

```bash
python verify.py
pytest -q
```

A successful `verify.py` run reports:

```text
PASS: 260 regenerated feature-cache hashes and 240 manifest hashes verified
PASS: deterministic synthetic cost cache verified
PASS: paired rotation protocol
PASS: fixed-test nested few-shot protocol
PASS: exact entropy definition
PASS: ground-cost and Sinkhorn unit checks
```

If you do not want `verify.py` to prepare missing caches automatically, use:

```bash
python verify.py --no-prepare
```

To regenerate every local cache before verification:

```bash
python verify.py --force-prepare
```

The frozen expected cache hashes are stored in `provenance/reference_data_metadata.json`. Cache generation does **not** rewrite that file or the committed manifests, so verification remains meaningful.

## Full reproduction

The recommended end-to-end command is:

```bash
python rerun_v4.py
```

It performs the following sequence:

1. reconstruct all local HOG/synthetic caches from the frozen v4 manifests;
2. verify generated cache hashes and protocol invariants;
3. run validation-only hyperparameter selection on Digits seeds `0--4`;
4. run all held-out experiments on seeds `5--19` using the selected hyperparameters;
5. generate derived summaries and statistical tests;
6. regenerate the figures.

The same workflow can be run manually:

```bash
python code/prepare_reference_data.py --root . --overwrite
python verify.py --no-prepare
python code/run_validation.py --root . --outdir results/validation_rerun
python code/run_experiments.py --root . --outdir results/rerun_full --selected-config results/validation_rerun/selected_hyperparameters.json
python code/analyze_results.py --root . --input-dir results/rerun_full --outdir results/derived_rerun
python code/generate_figures.py --root . --results-dir results/rerun_full --outdir figures_rerun
```

`--rebuild-protocol` on `prepare_reference_data.py` is a maintainer-only option that regenerates manifests and frozen reference metadata. Ordinary users should reproduce from the committed v4 manifests.

## Repository structure

```text
TAOT-SRC/
|-- code/                         # method, data preparation, validation, experiments, analysis
|-- data/
|   `-- manifests/                # deterministic v4 split/rotation manifests (tracked)
|-- figures/                      # manuscript topology figure
|-- figures_rerun/                # figures regenerated from the frozen v4 record
|-- provenance/                   # environment + frozen expected cache/manifest hashes
|-- results/
|   |-- validation_rerun/         # validation sweeps + selected hyperparameters
|   |-- rerun_full/               # per-seed authoritative v4 outputs
|   `-- derived_rerun/            # summaries and statistical analyses
|-- tests/                        # mathematical/protocol tests
|-- METHOD_DEFINITION.md          # exact method-to-code specification
|-- PROTOCOL_CORRECTION_V4.md     # v4 protocol correction details
|-- VERIFICATION_POLICY.md        # integrity and verification policy
|-- RESULTS.md                    # frozen v4 numerical summary
|-- MANIFEST.md                   # package contents
|-- SHA256SUMS.txt                # hashes of tracked GitHub-package files
|-- requirements.txt
|-- verify.py
`-- rerun_v4.py
```

After local data preparation, an ignored directory is created:

```text
data/features/                    # generated HOG/synthetic caches; not tracked
```

## Frozen implementation settings

- HOG: `2 x 2` spatial cells, 9 unsigned orientation bins, 36 bins total
- orientation-bin coordinate: `(b + 0.5) * pi / 9`
- orientation voting: hard single-bin assignment
- histogram floor: `1e-6`, followed by unit-mass normalization
- ground-cost denominator safeguard: `1e-12`
- Adam: 70 updates, learning rate `0.35`
- Sinkhorn: 30 iterations during TAOT optimization, 50 for TAOT final scoring, 60 for pairwise OT baselines
- signed SRC alpha: `1e-4`
- CRC ridge: `1e-2`
- RBF-SVM: `C=3`, `gamma=scale`
- validation seeds: `0--4`
- held-out seeds: `5--19`
- few-shot settings: `1, 2, 3, 5, 10` samples/class
- coefficient-log safeguard: `1e-9`

## Integrity

`SHA256SUMS.txt` contains hashes for the files distributed in the GitHub package. Generated files under `data/features/` are intentionally excluded from `SHA256SUMS.txt` and from Git.

For experiment-specific integrity, `provenance/reference_data_metadata.json` stores the frozen expected hashes for all generated feature conditions and committed manifests. After the local caches are prepared, run:

```bash
python verify.py
```

This checks the regenerated data against the frozen expected hashes rather than rewriting those expectations. To make this portable across platforms, feature verification hashes canonicalized numerical array content (float arrays rounded to 12 decimal places) rather than the binary `.npz` container bytes.

## Citation

If you use this code, please cite the associated TAOT-SRC manuscript. Until the final bibliographic record is available, the repository can be cited as:

```text
Shujaat Khan. Topology-Aware Optimal Transport for Sparse Representation Classification (TAOT-SRC), v4, 2026.
https://github.com/Shujaat123/TAOT-SRC
```

A machine-readable citation is also provided in [`CITATION.cff`](CITATION.cff).

## Notes

- No manuscript LaTeX is included in this repository.
- No raw image dataset is redistributed by this repository.
- No derived `data/features/` cache is redistributed by this repository.
- No v3 reference-result CSVs are included.
- v4 is the authoritative experimental record for this repository version.
