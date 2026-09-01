#!/usr/bin/env python3
"""TAOT-SRC v4 protocol verifier / rerun helper.

python verify.py                 # cache + protocol + mathematical checks
python verify.py --mode quick    # fresh one-seed smoke run
python verify.py --mode full     # full fresh rerun using selected config if supplied
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; CODE=ROOT/'code'

def run(cmd):
    print('\n$', ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['reference','quick','full'],default='reference')
    ap.add_argument('--outdir',default=None)
    ap.add_argument('--selected-config',default=None)
    args=ap.parse_args(); py=sys.executable
    run([py,CODE/'verify_reference.py','--root',ROOT])
    if args.mode=='reference': return
    out=Path(args.outdir or ('results/rerun_quick' if args.mode=='quick' else 'results/rerun_full'))
    out=out if out.is_absolute() else ROOT/out
    cmd=[py,CODE/'run_experiments.py','--root',ROOT,'--outdir',out]
    if args.mode=='quick': cmd.append('--quick')
    if args.selected_config: cmd += ['--selected-config',args.selected_config]
    run(cmd)
    print('\nFresh rerun completed successfully. v4 intentionally does not compare against v3 references.')
if __name__=='__main__': main()
