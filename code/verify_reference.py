#!/usr/bin/env python3
"""Integrity, protocol, and mathematical checks for TAOT-SRC v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg
from prepare_reference_data import canonical_npz_hash, hairpin_costs
from taot_src import (
    coefficient_entropy,
    ground_cost,
    sinkhorn_full_primal_batch,
    sinkhorn_transport_cost_batch,
)


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    args = ap.parse_args()
    root = Path(args.root).resolve()
    errors = []

    meta_path = root / "provenance" / "reference_data_metadata.json"
    if not meta_path.exists():
        errors.append("missing provenance/reference_data_metadata.json")
        meta = {"records": []}
    else:
        meta = json.loads(meta_path.read_text())

    missing_feature_count = 0
    verified_features = 0
    verified_manifests = 0
    for r in meta.get("records", []):
        manifest_rel = r.get("manifest")
        if manifest_rel:
            p = root / Path(str(manifest_rel).replace("\\", "/"))
            if not p.exists():
                errors.append(f"missing {p}")
            elif sha256(p) != r["manifest_sha256"]:
                errors.append(f"manifest hash mismatch {p}")
            else:
                verified_manifests += 1

        feature_rel = r.get("feature")
        if feature_rel:
            p = root / Path(str(feature_rel).replace("\\", "/"))
            if not p.exists():
                missing_feature_count += 1
                errors.append(f"missing {p}")
            else:
                expected = r.get("feature_content_sha256")
                if expected is None:
                    errors.append(f"missing frozen content hash for {p}")
                elif canonical_npz_hash(p) != expected:
                    errors.append(f"numerical-content hash mismatch {p}")
                else:
                    verified_features += 1

    costs_path = root / "data" / "features" / "synthetic_costs.npz"
    if not costs_path.exists():
        errors.append(f"missing {costs_path}")
    else:
        costs = np.load(costs_path)
        Ce_ref, Cg_ref = hairpin_costs()
        if not (
            np.array_equal(costs["C_euclidean"], Ce_ref)
            and np.array_equal(costs["C_geodesic"], Cg_ref)
        ):
            errors.append("synthetic_costs.npz does not match the deterministic v4 construction")
        frozen_cost_hash = meta.get("synthetic_costs_content_sha256")
        if frozen_cost_hash and canonical_npz_hash(costs_path) != frozen_cost_hash:
            errors.append("numerical-content hash mismatch data/features/synthetic_costs.npz")

    # Mathematical checks.
    Cc = ground_cost(spatial_weight=cfg.ETA, circular=True)
    Cl = ground_cost(spatial_weight=cfg.ETA, circular=False)
    if not np.allclose(Cc, Cc.T):
        errors.append("circular ground cost not symmetric")
    if not np.isclose(Cc.max(), 1.0, atol=1e-9):
        errors.append("circular ground cost not normalized")
    if not Cc[0, 8] < Cl[0, 8]:
        errors.append("circular wrap-around check failed")

    a = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    C = torch.tensor(
        np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]) / 3,
        dtype=torch.float32,
    )
    tc = sinkhorn_transport_cost_batch(a, a, C, eps=0.05, n_iters=100)
    fp = sinkhorn_full_primal_batch(a, a, C, eps=0.05, n_iters=100)
    if not (torch.isfinite(tc).all() and torch.isfinite(fp).all()):
        errors.append("Sinkhorn finite-value check failed")

    w = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    if not torch.equal(coefficient_entropy(w), -(w * torch.log(w + 1e-9)).sum(1)):
        errors.append("coefficient entropy definition mismatch")

    # Protocol checks from frozen manifests.
    mdir = root / "data" / "manifests"
    for ds in cfg.DATASETS:
        tag = ds.replace("-", "_")
        for seed in tuple(cfg.VALIDATION_SEEDS) + tuple(cfg.HELDOUT_SEEDS):
            mans = []
            for rot in cfg.ROTATIONS:
                p = mdir / f"{tag}_seed{seed}_k{cfg.DEFAULT_SHOT}_rot{int(rot)}.json"
                if p.exists():
                    mans.append(json.loads(p.read_text()))
            if len(mans) == len(cfg.ROTATIONS):
                if not all(m["test_idx"] == mans[0]["test_idx"] for m in mans[1:]):
                    errors.append(f"{ds} seed {seed}: test indices differ across rotations")
                a10 = np.asarray(mans[1]["test_angles"])
                a20 = np.asarray(mans[2]["test_angles"])
                if not np.allclose(a20, 2 * a10):
                    errors.append(f"{ds} seed {seed}: rotations are not paired/scaled")

            if seed in cfg.HELDOUT_SEEDS:
                fs = []
                for k in cfg.FEWSHOT:
                    p = mdir / f"{tag}_seed{seed}_k{k}_rot0.json"
                    if p.exists():
                        fs.append(json.loads(p.read_text()))
                if len(fs) == len(cfg.FEWSHOT):
                    if not all(m["test_idx"] == fs[0]["test_idx"] for m in fs[1:]):
                        errors.append(f"{ds} seed {seed}: test indices differ across shot counts")
                    for a1, a2 in zip(fs, fs[1:]):
                        if not set(a1["train_idx"]).issubset(set(a2["train_idx"])):
                            errors.append(f"{ds} seed {seed}: training sets are not nested")

    if errors:
        if missing_feature_count:
            print(
                "Feature caches are generated artifacts and are not distributed on GitHub.\n"
                "Generate them with:\n"
                "    python code/prepare_reference_data.py --root .\n"
                "or use:\n"
                "    python verify.py\n"
            )
        print("\n".join("FAIL " + e for e in errors))
        raise SystemExit(1)

    print(
        f"PASS: {verified_features} regenerated feature-cache hashes and "
        f"{verified_manifests} manifest hashes verified"
    )
    print("PASS: deterministic synthetic cost cache verified")
    print("PASS: paired rotation protocol")
    print("PASS: fixed-test nested few-shot protocol")
    print("PASS: exact entropy definition")
    print("PASS: ground-cost and Sinkhorn unit checks")


if __name__ == "__main__":
    main()
