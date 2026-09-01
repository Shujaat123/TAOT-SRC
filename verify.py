#!/usr/bin/env python3
"""TAOT-SRC v4 verifier and rerun helper.

The public GitHub repository does not ship ``data/features``. On the first
verification, missing feature caches are regenerated automatically from the
frozen v4 manifests and the public dataset interfaces.

Examples
--------
python verify.py                 # prepare missing caches, then verify
python verify.py --no-prepare    # fail instead of auto-preparing missing caches
python verify.py --mode quick    # verify, then run a fresh one-seed smoke test
python verify.py --mode full     # verify, then run the full held-out experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def expected_feature_paths():
    meta_path = ROOT / "provenance" / "reference_data_metadata.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    out = []
    for rec in meta.get("records", []):
        rel = rec.get("feature")
        if rel:
            out.append(ROOT / Path(str(rel).replace("\\", "/")))
    out.append(ROOT / "data" / "features" / "synthetic_costs.npz")
    return out


def missing_features():
    return [p for p in expected_feature_paths() if not p.exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "quick", "full"], default="reference")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--selected-config", default=None)
    ap.add_argument(
        "--no-prepare",
        action="store_true",
        help="do not automatically regenerate missing data/features caches",
    )
    ap.add_argument(
        "--force-prepare",
        action="store_true",
        help="regenerate all local feature caches before verification",
    )
    args = ap.parse_args()
    py = sys.executable

    missing = missing_features()
    if args.force_prepare:
        print("Regenerating local feature caches from frozen v4 manifests.")
        run([py, CODE / "prepare_reference_data.py", "--root", ROOT, "--overwrite"])
    elif missing:
        if args.no_prepare:
            print(
                f"ERROR: {len(missing)} required generated feature files are missing.\n"
                "The public repository intentionally does not ship data/features.\n"
                "Run:\n"
                "    python code/prepare_reference_data.py --root .\n"
                "or simply rerun:\n"
                "    python verify.py\n",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(
            f"Feature caches are not present ({len(missing)} generated files missing).\n"
            "They are intentionally excluded from GitHub and will now be rebuilt\n"
            "from the frozen v4 manifests using scikit-learn/scikit-image data."
        )
        run([py, CODE / "prepare_reference_data.py", "--root", ROOT])

    run([py, CODE / "verify_reference.py", "--root", ROOT])

    if args.mode == "reference":
        return

    out = Path(
        args.outdir
        or ("results/rerun_quick" if args.mode == "quick" else "results/rerun_full")
    )
    out = out if out.is_absolute() else ROOT / out
    cmd = [py, CODE / "run_experiments.py", "--root", ROOT, "--outdir", out]
    if args.mode == "quick":
        cmd.append("--quick")
    if args.selected_config:
        cmd += ["--selected-config", args.selected_config]
    run(cmd)
    print("\nFresh rerun completed successfully. v4 is not compared against v3 references.")


if __name__ == "__main__":
    main()
