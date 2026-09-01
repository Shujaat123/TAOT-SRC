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
