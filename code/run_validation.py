#!/usr/bin/env python3
"""Validation-only hyperparameter sweeps on Digits seeds 0--4."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import config as cfg
from taot_src import taot_scores


def load(root,seed,rot):
 z=np.load(Path(root)/'data'/'features'/f'Digits_seed{seed}_k5_rot{rot}.npz')
 return z['Xtrain'],z['ytrain'],z['Xtest'],z['ytest']

def run_one(root,seed,rot,eta,eps,lam):
 Xtr,ytr,Xte,yte=load(root,seed,rot)
 s,c=taot_scores(Xte,Xtr,ytr,eps=eps,lam=lam,spatial_weight=eta,circular=True,
                 steps=cfg.ADAM_STEPS,lr=cfg.ADAM_LR,opt_sinkhorn_iters=cfg.SINKHORN_OPT_ITERS,
                 final_sinkhorn_iters=cfg.SINKHORN_FINAL_ITERS)
 return accuracy_score(yte,c[s.argmin(1)])

def _best(summary, param):
    # deterministic tie break: smaller parameter value
    s=summary.sort_values(['accuracy',param],ascending=[False,True]).reset_index(drop=True)
    return float(s.loc[0,param])

def main():
    import json
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(HERE.parent)); ap.add_argument('--outdir',default='results/validation_rerun'); args=ap.parse_args()
    root=Path(args.root).resolve(); out=Path(args.outdir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
    rows=[]

    # Sequential validation-only coordinate sweep.  No held-out seed is used.
    eta0=float(cfg.ETA); eps0=float(cfg.EPSILON)

    for lam in cfg.LAMBDA_GRID:
      for rot in cfg.ROTATIONS:
       for seed in cfg.VALIDATION_SEEDS:
        rows.append({'sweep':'lambda','lambda':lam,'epsilon':eps0,'eta':eta0,'rotation':rot,'seed':seed,'accuracy':run_one(root,seed,rot,eta0,eps0,lam)})
    d1=pd.DataFrame(rows)
    g1=d1[d1.sweep=='lambda'].groupby('lambda',as_index=False).accuracy.mean()
    lam_sel=_best(g1,'lambda')

    rows2=[]
    for eps in cfg.EPSILON_GRID:
      for rot in cfg.ROTATIONS:
       for seed in cfg.VALIDATION_SEEDS:
        rows2.append({'sweep':'epsilon','lambda':lam_sel,'epsilon':eps,'eta':eta0,'rotation':rot,'seed':seed,'accuracy':run_one(root,seed,rot,eta0,eps,lam_sel)})
    d2=pd.DataFrame(rows2); g2=d2.groupby('epsilon',as_index=False).accuracy.mean(); eps_sel=_best(g2,'epsilon')

    rows3=[]
    for eta in cfg.ETA_GRID:
      for rot in cfg.ROTATIONS:
       for seed in cfg.VALIDATION_SEEDS:
        rows3.append({'sweep':'eta','lambda':lam_sel,'epsilon':eps_sel,'eta':eta,'rotation':rot,'seed':seed,'accuracy':run_one(root,seed,rot,eta,eps_sel,lam_sel)})
    d3=pd.DataFrame(rows3); g3=d3.groupby('eta',as_index=False).accuracy.mean(); eta_sel=_best(g3,'eta')

    d=pd.concat([d1,d2,d3],ignore_index=True); d.to_csv(out/'validation_sensitivity.csv',index=False)
    g1.to_csv(out/'lambda_sensitivity_summary.csv',index=False)
    g2.to_csv(out/'epsilon_sensitivity_summary.csv',index=False)
    g3.to_csv(out/'eta_sensitivity_summary.csv',index=False)
    selected={'lambda':lam_sel,'epsilon':eps_sel,'eta':eta_sel,'selection_protocol':'sequential validation-only sweeps on Digits seeds 0-4'}
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    print('Selected:',selected)
    print('Wrote validation sweep to',out)

if __name__=='__main__': main()
