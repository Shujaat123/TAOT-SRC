# TAOT-SRC v4 GitHub package contents

- `code/` -- executable method, public-data preparation, validation, experiments, analysis, runtime, and figure scripts.
- `data/manifests/` -- corrected deterministic split and paired-rotation manifests.
- `results/validation_rerun/` -- v4 validation sweeps and selected hyperparameters.
- `results/rerun_full/` -- authoritative per-seed v4 experimental outputs.
- `results/derived_rerun/` -- frozen summaries and statistical analyses.
- `figures/` -- topology figure used by the manuscript.
- `figures_rerun/` -- regenerated v4 experimental figures.
- `tests/` -- mathematical and protocol unit tests.
- `provenance/reference_data_metadata.json` -- frozen expected manifest/feature hashes and generation metadata.
- `verify.py` -- first-run data preparation plus integrity/protocol verifier.
- `rerun_v4.py` -- recommended end-to-end rerun entry point.
- `METHOD_DEFINITION.md` -- exact method-to-code definition.
- `PROTOCOL_CORRECTION_V4.md` -- explanation of the corrected v4 evaluation protocol.
- `VERIFICATION_POLICY.md` -- integrity and verification policy.
- `RESULTS.md` -- frozen v4 numerical summary.
- `requirements.txt` -- Python dependencies.
- `SHA256SUMS.txt` -- hashes of distributed GitHub-package files.
- `CITATION.cff` -- machine-readable software citation.

## Data not distributed

The public GitHub package intentionally does **not** contain `data/features/`.
Those HOG/synthetic caches are generated locally from the committed manifests
and the public scikit-learn/scikit-image dataset interfaces by:

```bash
python code/prepare_reference_data.py --root .
```

or automatically by:

```bash
python verify.py
```

`data/features/` is listed in `.gitignore` and is not included in
`SHA256SUMS.txt`. Frozen expected hashes for the generated feature conditions
remain in `provenance/reference_data_metadata.json` so regenerated caches can
be checked exactly.

Generated Python bytecode, pytest caches, virtual environments, editor/OS
temporary files, and locally generated feature caches are excluded from the
GitHub package.

No manuscript LaTeX and no v3 numerical reference CSVs are included.
