# TAOT-SRC v4 GitHub package contents

- `code/` -- executable method, preprocessing, validation, experiments, analysis, runtime, and figure scripts.
- `data/manifests/` -- corrected deterministic split and paired-rotation manifests.
- `data/features/` -- HOG and synthetic caches generated from the corrected manifests.
- `results/validation_rerun/` -- v4 validation sweeps and selected hyperparameters.
- `results/rerun_full/` -- authoritative per-seed v4 experimental outputs.
- `results/derived_rerun/` -- frozen summaries and statistical analyses.
- `figures/` -- topology figure used by the manuscript.
- `figures_rerun/` -- regenerated v4 experimental figures.
- `tests/` -- mathematical and protocol unit tests.
- `provenance/reference_data_metadata.json` -- cache hashes and generation metadata.
- `verify.py` -- cached-condition/protocol verifier.
- `rerun_v4.py` -- recommended end-to-end rerun entry point.
- `METHOD_DEFINITION.md` -- exact method-to-code definition.
- `PROTOCOL_CORRECTION_V4.md` -- explanation of the corrected v4 evaluation protocol.
- `VERIFICATION_POLICY.md` -- integrity and verification policy.
- `RESULTS.md` -- frozen v4 numerical summary.
- `requirements.txt` -- Python dependencies.
- `SHA256SUMS.txt` -- GitHub package file hashes.
- `CITATION.cff` -- machine-readable software citation.

Generated Python bytecode, pytest caches, virtual environments, and editor/OS temporary files are excluded from the GitHub package.

No manuscript LaTeX and no v3 numerical reference CSVs are included.
