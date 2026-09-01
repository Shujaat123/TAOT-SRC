# Migration notes: v3 -> v4

v4 is a protocol correction, not merely a verifier update.

- rotation conditions now share the same train/test indices for a seed;
- rotation RNG is independent of split RNG;
- +/-10 and +/-20 degree angles are paired by scaling one `Uniform(-1,1)` draw per test sample;
- the previous small-angle remapping rule has been removed;
- few-shot experiments now use one fixed test set and nested training subsets;
- coefficient entropy is implemented exactly as `-sum w log(w + 1e-9)`;
- validation writes `selected_hyperparameters.json` and held-out experiments can consume it with `--selected-config`;
- v3 result CSVs are intentionally not shipped as v4 reference targets.

Because the experimental sample/perturbation protocol changed, all reported v4 tables, statistics, figures, and manuscript text must be generated from the new rerun.

## Public GitHub data packaging

The public v4 GitHub repository does not redistribute `data/features/`.
Derived HOG/synthetic caches are regenerated locally from the frozen v4
manifests and public dataset interfaces. `python verify.py` performs this
preparation automatically when the cache directory is absent.

Because `.npz` container bytes are not a portable reproducibility target,
feature verification uses frozen SHA256 hashes of canonicalized numerical
array content (float arrays rounded to 12 decimal places) rather than hashes
of the ZIP container bytes. Committed manifest JSON files continue to use
ordinary file SHA256 hashes.
