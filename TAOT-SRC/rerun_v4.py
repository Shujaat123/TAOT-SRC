#!/usr/bin/env python3
"""End-to-end v4 rerun: rebuild caches -> validate -> held-out experiments -> analysis -> figures."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent; CODE=ROOT/'code'; PY=sys.executable

def run(cmd):
    print('\n$', ' '.join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)

def main():
    run([PY,CODE/'prepare_reference_data.py','--root',ROOT,'--overwrite'])
    run([PY,CODE/'verify_reference.py','--root',ROOT])
    val=ROOT/'results'/'validation_rerun'
    run([PY,CODE/'run_validation.py','--root',ROOT,'--outdir',val])
    selected=val/'selected_hyperparameters.json'
    full=ROOT/'results'/'rerun_full'
    run([PY,CODE/'run_experiments.py','--root',ROOT,'--outdir',full,'--selected-config',selected])
    derived=ROOT/'results'/'derived_rerun'
    run([PY,CODE/'analyze_results.py','--root',ROOT,'--input-dir',full,'--outdir',derived])
    figs=ROOT/'figures_rerun'
    run([PY,CODE/'generate_figures.py','--root',ROOT,'--results-dir',full,'--outdir',figs])
    print('\nV4 rerun complete.')
    print('Selected hyperparameters:',json.loads(selected.read_text()))
    print('Results:',full)
    print('Derived:',derived)
    print('Figures:',figs)
if __name__=='__main__': main()
