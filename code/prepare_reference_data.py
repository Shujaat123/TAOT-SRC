#!/usr/bin/env python3
"""Prepare TAOT-SRC v4 feature caches from public data.

Public-repository default behavior
----------------------------------
The GitHub repository intentionally does not redistribute ``data/features``.
This script reconstructs the feature caches from the frozen v4 manifests and
public dataset interfaces while preserving the committed manifests and frozen
reference metadata.

Use:
    python code/prepare_reference_data.py --root .
    python code/prepare_reference_data.py --root . --overwrite

Maintainer-only protocol regeneration
-------------------------------------
``--rebuild-protocol`` regenerates manifests and reference metadata from the
protocol code. Ordinary reproduction should not need this option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from sklearn.datasets import load_digits
from skimage import data

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg
from taot_src import hog_hist


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_npz_hash(path: Path, decimals: int = 12):
    """Platform-stable SHA256 of numerical NPZ content, not ZIP container bytes."""
    h = hashlib.sha256()
    with np.load(path) as z:
        for key in sorted(z.files):
            a = np.asarray(z[key])
            if a.dtype.kind in "fc":
                a = np.round(a.astype(np.float64), decimals=decimals)
                dtype_tag = f"float64-rounded-{decimals}"
            elif a.dtype.kind in "iu":
                a = a.astype(np.int64)
                dtype_tag = "int64"
            elif a.dtype.kind == "b":
                a = a.astype(np.uint8)
                dtype_tag = "uint8-bool"
            else:
                raise TypeError(f"Unsupported array dtype in {path}: {key} -> {a.dtype}")
            a = np.ascontiguousarray(a)
            h.update(key.encode("utf-8") + b"\0")
            h.update(dtype_tag.encode("ascii") + b"\0")
            h.update(",".join(map(str, a.shape)).encode("ascii") + b"\0")
            h.update(a.tobytes(order="C"))
    return h.hexdigest()


def datasets():
    """Load the two real datasets through their maintained Python interfaces."""
    ds = load_digits()
    digits = (ds.images.astype(float), ds.target.astype(int))

    # scikit-image distributes the exact 200-image LFW face/non-face subset
    # used by the paper (100 face + 100 non-face images).
    lfw = data.lfw_subset().astype(float)
    lfw_y = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    return {"Digits": digits, "LFW-Face": (lfw, lfw_y)}


def build_manifest(y, seed, k, ntest, rotation, max_k=None):
    """Build a paired split/rotation manifest according to protocol v4.

    This function is retained for unit tests and maintainer-side protocol
    regeneration. Public users normally consume the frozen manifests already
    committed under ``data/manifests``.
    """
    if max_k is None:
        max_k = cfg.MAX_SHOT
    if k > max_k:
        raise ValueError(f"k={k} exceeds max_k={max_k}")

    split_rng = np.random.default_rng(seed)
    train_idx, test_idx, ytr, yte = [], [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        split_rng.shuffle(idx)
        pool = idx[:max_k]
        tei = idx[max_k:max_k + ntest]
        tri = pool[:k]
        train_idx.extend(map(int, tri))
        ytr.extend([int(c)] * len(tri))
        test_idx.extend(map(int, tei))
        yte.extend([int(c)] * len(tei))

    angle_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 424242]))
    factors = angle_rng.uniform(-1.0, 1.0, size=len(test_idx))
    test_angles = (
        (factors * float(rotation)).astype(float).tolist()
        if rotation
        else [0.0] * len(test_idx)
    )

    return dict(
        seed=int(seed),
        k=int(k),
        max_k=int(max_k),
        ntest=int(ntest),
        rotation=float(rotation),
        train_idx=train_idx,
        test_idx=test_idx,
        test_angles=test_angles,
        ytrain=ytr,
        ytest=yte,
    )


def feature_from_manifest(images, man):
    tr = images[np.asarray(man["train_idx"], int)]
    te = []
    for ii, ang in zip(man["test_idx"], man["test_angles"]):
        im = images[int(ii)]
        if ang:
            im = ndi_rotate(im, ang, reshape=False, order=1, mode="constant", cval=0)
        te.append(im)

    Xtr = np.asarray(
        [hog_hist(im, cells=cfg.CELLS, nbins=cfg.NBINS, floor=cfg.HOG_FLOOR) for im in tr]
    )
    Xte = np.asarray(
        [hog_hist(im, cells=cfg.CELLS, nbins=cfg.NBINS, floor=cfg.HOG_FLOOR) for im in te]
    )
    return Xtr, np.asarray(man["ytrain"], int), Xte, np.asarray(man["ytest"], int)


def gaussian_hist(center_idx, n=40, sigma=1.8):
    idx = np.arange(n)
    h = np.exp(-0.5 * ((idx - center_idx) / sigma) ** 2)
    return h / h.sum()


def synthetic_split(seed, ntrain=4, ntest=50, noise=0.02):
    rng = np.random.default_rng(seed)
    centers = [8, 31, 20]
    Xtr, ytr, Xte, yte = [], [], [], []
    for c, mu in enumerate(centers):
        for _ in range(ntrain):
            h = gaussian_hist(mu + rng.normal(0, 1))
            h = np.maximum(h + rng.normal(0, noise, 40), 0)
            h /= h.sum()
            Xtr.append(h)
            ytr.append(c)
        for _ in range(ntest):
            ctr = np.clip(mu + rng.normal(0, 6), 0, 39)
            h = gaussian_hist(ctr)
            h = np.maximum(h + rng.normal(0, noise, 40), 0)
            h /= h.sum()
            Xte.append(h)
            yte.append(c)
    return np.asarray(Xtr), np.asarray(ytr), np.asarray(Xte), np.asarray(yte)


def hairpin_costs(n_arm=18, n_conn=4, gap=0.12):
    x1 = np.linspace(0, 1, n_arm)
    y1 = np.zeros(n_arm)
    yc = np.linspace(0, gap, n_conn + 2)[1:-1]
    xc = np.ones(n_conn)
    x2 = np.linspace(1, 0, n_arm)
    y2 = np.full(n_arm, gap)
    coords = np.c_[np.r_[x1, xc, x2], np.r_[y1, yc, y2]]
    ds = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(1))
    s = np.r_[0, np.cumsum(ds)]
    Ce = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    Cg = (s[:, None] - s[None, :]) ** 2
    return Ce / Ce.max(), Cg / Cg.max()


def frozen_metadata(root: Path):
    p = root / "provenance" / "reference_data_metadata.json"
    if not p.exists():
        raise FileNotFoundError(
            "Missing provenance/reference_data_metadata.json. "
            "The public package requires the frozen metadata file."
        )
    return p, json.loads(p.read_text())


def prepare_from_frozen_manifests(root: Path, overwrite: bool = False):
    """Generate local feature caches without altering frozen manifests/metadata."""
    mdir = root / "data" / "manifests"
    fdir = root / "data" / "features"
    if not mdir.exists():
        raise FileNotFoundError(
            "Missing data/manifests. Restore the committed v4 manifests before preparing data."
        )
    fdir.mkdir(parents=True, exist_ok=True)

    _, meta = frozen_metadata(root)
    dsets = datasets()
    generated = skipped = 0

    for rec in meta.get("records", []):
        feature_rel = rec.get("feature")
        if not feature_rel:
            continue
        fp = root / Path(str(feature_rel).replace("\\", "/"))
        if fp.exists() and not overwrite:
            skipped += 1
            continue
        fp.parent.mkdir(parents=True, exist_ok=True)

        ds_name = rec.get("dataset")
        if ds_name in dsets:
            manifest_rel = rec.get("manifest")
            if not manifest_rel:
                raise RuntimeError(f"Missing manifest entry for {feature_rel}")
            mp = root / Path(str(manifest_rel).replace("\\", "/"))
            if not mp.exists():
                raise FileNotFoundError(f"Missing frozen manifest: {mp}")
            man = json.loads(mp.read_text())
            images, _ = dsets[ds_name]
            Xtr, ytr, Xte, yte = feature_from_manifest(images, man)
        elif ds_name == "Synthetic-Hairpin":
            Xtr, ytr, Xte, yte = synthetic_split(int(rec["seed"]))
        else:
            raise RuntimeError(f"Unknown dataset in frozen metadata: {ds_name!r}")

        np.savez_compressed(fp, Xtrain=Xtr, ytrain=ytr, Xtest=Xte, ytest=yte)
        generated += 1

    costs_path = fdir / "synthetic_costs.npz"
    if overwrite or not costs_path.exists():
        Ce, Cg = hairpin_costs()
        np.savez_compressed(costs_path, C_euclidean=Ce, C_geodesic=Cg)

    print(
        f"Prepared feature caches under {fdir} "
        f"({generated} generated, {skipped} already present)."
    )
    print("Frozen manifests and provenance/reference_data_metadata.json were not modified.")


def rebuild_protocol(root: Path, overwrite: bool = False):
    """Maintainer-only regeneration of manifests, features, and frozen metadata."""
    mdir = root / "data" / "manifests"
    fdir = root / "data" / "features"
    pdir = root / "provenance"
    mdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    dsets = datasets()
    records = []
    jobs = []

    for ds in cfg.DATASETS:
        ntest = cfg.DIGITS_NTEST if ds == "Digits" else cfg.LFW_NTEST
        for seed in cfg.HELDOUT_SEEDS:
            for rot in cfg.ROTATIONS:
                jobs.append((ds, seed, cfg.DEFAULT_SHOT, ntest, rot, "rotation"))
            for k in cfg.FEWSHOT:
                jobs.append((ds, seed, k, ntest, 0, "fewshot"))
        for seed in cfg.VALIDATION_SEEDS:
            for rot in cfg.ROTATIONS:
                jobs.append((ds, seed, cfg.DEFAULT_SHOT, ntest, rot, "validation"))

    seen = set()
    for ds, seed, k, ntest, rot, purpose in jobs:
        key = (ds, seed, k, rot)
        if key in seen:
            continue
        seen.add(key)
        images, y = dsets[ds]
        man = build_manifest(y, seed, k, ntest, rot)
        stem = f"{ds.replace('-', '_')}_seed{seed}_k{k}_rot{int(rot)}"
        mp = mdir / f"{stem}.json"
        fp = fdir / f"{stem}.npz"
        if fp.exists() and not overwrite:
            raise FileExistsError(f"{fp} exists; use --overwrite for protocol rebuild")
        mp.write_text(json.dumps({"dataset": ds, "purpose": purpose, **man}, indent=2))
        Xtr, ytr, Xte, yte = feature_from_manifest(images, man)
        np.savez_compressed(fp, Xtrain=Xtr, ytrain=ytr, Xtest=Xte, ytest=yte)
        records.append(
            {
                "dataset": ds,
                "seed": seed,
                "k": k,
                "rotation": rot,
                "manifest": mp.relative_to(root).as_posix(),
                "feature": fp.relative_to(root).as_posix(),
                "manifest_sha256": sha256(mp),
                "feature_content_sha256": canonical_npz_hash(fp),
                "ntrain": len(ytr),
                "ntest": len(yte),
            }
        )

    Ce, Cg = hairpin_costs()
    costs_path = fdir / "synthetic_costs.npz"
    np.savez_compressed(costs_path, C_euclidean=Ce, C_geodesic=Cg)

    for seed in range(20):
        Xtr, ytr, Xte, yte = synthetic_split(seed)
        fp = fdir / f"synthetic_seed{seed}.npz"
        np.savez_compressed(fp, Xtrain=Xtr, ytrain=ytr, Xtest=Xte, ytest=yte)
        records.append(
            {
                "dataset": "Synthetic-Hairpin",
                "seed": seed,
                "k": 4,
                "rotation": 0,
                "feature": fp.relative_to(root).as_posix(),
                "feature_content_sha256": canonical_npz_hash(fp),
                "ntrain": len(ytr),
                "ntest": len(yte),
            }
        )

    meta = {
        "python": sys.version,
        "config": {
            k: getattr(cfg, k)
            for k in [
                "CELLS",
                "NBINS",
                "HOG_FLOOR",
                "ETA",
                "EPSILON",
                "LAMBDA",
                "ADAM_STEPS",
                "ADAM_LR",
                "SINKHORN_OPT_ITERS",
                "SINKHORN_FINAL_ITERS",
                "SINKHORN_PAIR_ITERS",
                "SRC_ALPHA",
                "CRC_RIDGE",
                "SVM_C",
                "VALIDATION_SEEDS",
                "HELDOUT_SEEDS",
                "ROTATIONS",
                "FEWSHOT",
            ]
        },
        "synthetic_costs_content_sha256": canonical_npz_hash(costs_path),
        "feature_content_hash_policy": {
            "algorithm": "sha256",
            "array_order": "sorted NPZ keys; C-order bytes",
            "float_canonicalization": "cast float arrays to float64 and round to 12 decimal places",
            "integer_canonicalization": "cast integer arrays to int64",
            "purpose": "platform-stable verification of regenerated numerical cache content; NPZ container bytes are not hashed",
        },
        "records": records,
    }
    (pdir / "reference_data_metadata.json").write_text(
        json.dumps(meta, indent=2, default=list)
    )
    print(f"Rebuilt {len(records)} cached conditions and frozen metadata under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite locally generated feature caches",
    )
    ap.add_argument(
        "--rebuild-protocol",
        action="store_true",
        help="maintainer only: regenerate manifests and frozen reference metadata",
    )
    args = ap.parse_args()
    root = Path(args.root).resolve()

    if args.rebuild_protocol:
        rebuild_protocol(root, overwrite=args.overwrite)
    else:
        prepare_from_frozen_manifests(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
