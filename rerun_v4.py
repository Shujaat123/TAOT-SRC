#!/usr/bin/env python3
"""End-to-end v4 rerun from a public GitHub checkout.

Generated feature caches are reconstructed from the frozen v4 manifests and
public dataset interfaces, then validation, held-out experiments, analysis,
and figures are rerun.
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
PYTHON = sys.executable


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def main():
    # data/features is intentionally absent from the GitHub repository.
    # Rebuild local derived caches without modifying frozen manifests/metadata.
    run([PYTHON, CODE / "prepare_reference_data.py", "--root", ROOT, "--overwrite"])
    run([PYTHON, CODE / "verify_reference.py", "--root", ROOT])

    val = ROOT / "results" / "validation_rerun"
    run([PYTHON, CODE / "run_validation.py", "--root", ROOT, "--outdir", val])
    selected = val / "selected_hyperparameters.json"

    full = ROOT / "results" / "rerun_full"
    run(
        [
            PYTHON,
            CODE / "run_experiments.py",
            "--root",
            ROOT,
            "--outdir",
            full,
            "--selected-config",
            selected,
        ]
    )

    derived = ROOT / "results" / "derived_rerun"
    run(
        [
            PYTHON,
            CODE / "analyze_results.py",
            "--root",
            ROOT,
            "--input-dir",
            full,
            "--outdir",
            derived,
        ]
    )

    figs = ROOT / "figures_rerun"
    run(
        [
            PYTHON,
            CODE / "generate_figures.py",
            "--root",
            ROOT,
            "--results-dir",
            full,
            "--outdir",
            figs,
        ]
    )

    print("\nV4 rerun complete.")
    print("Selected hyperparameters:", json.loads(selected.read_text()))
    print("Results:", full)
    print("Derived:", derived)
    print("Figures:", figs)


if __name__ == "__main__":
    main()
