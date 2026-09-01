#!/usr/bin/env python3
"""Compare fresh per-seed outputs against the shipped authoritative reference.

All reported classification outputs use a tight default tolerance. The
continuous ``GS-effective-atoms`` diagnostic exists only in the few-shot
files and is allowed a small cross-platform tolerance because it is produced
by float32 Adam/Sinkhorn optimization. Rotation files intentionally omit this
unreported diagnostic.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

FILES = [
    'heldout_rotation_Digits.csv',
    'heldout_rotation_LFW-Face.csv',
    'heldout_fewshot_Digits.csv',
    'heldout_fewshot_LFW-Face.csv',
    'synthetic_hairpin.csv',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--atol', type=float, default=1e-7)
    ap.add_argument('--effective-atoms-atol', type=float, default=5e-4)
    args = ap.parse_args()

    overall_ok = True
    for name in FILES:
        a = pd.read_csv(Path(args.reference) / name)
        b = pd.read_csv(Path(args.candidate) / name)
        file_ok = True

        if list(a.columns) != list(b.columns) or len(a) != len(b):
            print('FAIL structure', name)
            overall_ok = file_ok = False
            continue

        for c in a.columns:
            if np.issubdtype(a[c].dtype, np.number):
                atol = (args.effective_atoms_atol if (name.startswith('heldout_fewshot_') and c == 'GS-effective-atoms') else args.atol)
                av = a[c].to_numpy()
                bv = b[c].to_numpy()
                if not np.allclose(av, bv, atol=atol, rtol=0, equal_nan=True):
                    maxdiff = float(np.nanmax(np.abs(av - bv)))
                    print('FAIL numeric', name, c, 'maxdiff', maxdiff, 'atol', atol)
                    overall_ok = file_ok = False
            elif not a[c].equals(b[c]):
                print('FAIL text', name, c)
                overall_ok = file_ok = False

        if file_ok:
            print('PASS', name)

    raise SystemExit(0 if overall_ok else 1)


if __name__ == '__main__':
    main()
