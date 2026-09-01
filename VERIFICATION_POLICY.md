# v4 verification policy

The protocol-corrected v4 experimental record is frozen in this repository.
The public GitHub repository intentionally does not redistribute the derived
`data/features/` cache files.

## First-run behavior

`python verify.py` checks whether the generated feature caches are present.
If they are absent, it automatically reconstructs them from:

- the committed v4 manifests in `data/manifests/`;
- `sklearn.datasets.load_digits()`;
- `skimage.data.lfw_subset()`;
- the deterministic synthetic-data generator in the repository.

The generated caches are local artifacts and remain ignored by Git.

Importantly, normal cache preparation does **not** rewrite the committed
manifests or `provenance/reference_data_metadata.json`. Therefore the frozen
expected hashes remain independent verification targets.

## What `python verify.py` checks

After any missing caches are generated, verification checks:

- frozen hashes for all generated feature conditions;
- frozen hashes for all committed real-data manifests;
- the deterministic synthetic ground-cost cache;
- identical test indices across clean / +/-10 / +/-20 conditions;
- paired angle scaling between +/-10 and +/-20;
- identical test sets across all few-shot settings;
- nested few-shot training sets;
- the exact coefficient-entropy definition;
- circular ground-cost properties;
- finite Sinkhorn calculations.

Use `python verify.py --no-prepare` to fail instead of generating missing
feature caches. Use `python verify.py --force-prepare` to rebuild all local
feature caches before checking them.

`pytest -q` provides independent unit tests for the key mathematical and
protocol properties.

## Numerical record

The authoritative frozen numerical outputs are stored in
`results/rerun_full/` and `results/derived_rerun/`. They are v4 outputs and
must not be compared against v3 values as numerical reference targets.

## File-level integrity

`SHA256SUMS.txt` contains hashes only for files distributed in the GitHub
package. Generated `data/features/` files are deliberately excluded.
Experiment-specific expected hashes for regenerated feature conditions and
committed manifests are stored in `provenance/reference_data_metadata.json`
and checked by `python verify.py`.
