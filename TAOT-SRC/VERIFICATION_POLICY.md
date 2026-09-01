# v4 verification policy

The protocol-corrected v4 experimental record is frozen in this repository. Verification therefore has two complementary purposes: protect the cached experimental conditions and test the protocol/mathematical invariants independently of the frozen result summaries.

`python verify.py` checks:

- hashes of all 260 shipped cached conditions;
- identical test indices across clean / +/-10 / +/-20 conditions for every available seed;
- paired angle scaling between +/-10 and +/-20;
- identical test sets across all few-shot settings;
- nested few-shot training sets;
- the exact coefficient-entropy definition;
- circular ground-cost properties;
- finite Sinkhorn calculations.

`pytest -q` provides independent unit tests for the key mathematical and protocol properties.

The authoritative frozen numerical outputs are stored in `results/rerun_full/` and `results/derived_rerun/`. They are v4 outputs and must not be compared against v3 values as numerical reference targets.

`SHA256SUMS.txt` provides file-level integrity hashes for the GitHub package. Experiment-specific cached-condition hashes are stored in `provenance/reference_data_metadata.json` and checked by `python verify.py`.
