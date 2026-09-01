#!/usr/bin/env python3
"""Runtime benchmark on fixed Digits seed-5 five-shot clean feature cache."""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import config as cfg
from taot_src import ground_cost, taot_scores
from experiment_utils import src_scores,posthoc_ot_scores,ot1nn_predict


def load(root):
 z=np.load(Path(root)/'data'/'features'/'Digits_seed5_k5_rot0.npz')
 return z['Xtrain'],z['ytrain'],z['Xtest'],z['ytest']

def timed(fn,repeats):
 xs=[]
 for _ in range(repeats):
  t=time.perf_counter(); fn(); xs.append(time.perf_counter()-t)
 return np.asarray(xs)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(HERE.parent)); ap.add_argument('--outdir',default='results/runtime_rerun'); ap.add_argument('--repeats',type=int,default=5); args=ap.parse_args()
 root=Path(args.root).resolve(); out=Path(args.outdir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
 Xtr,ytr,Xte,yte=load(root); Cc=ground_cost(spatial_weight=cfg.ETA,circular=True)
 def src(): src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=False)
 def circ_post():
  _,recs,_=src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=True)
  posthoc_ot_scores(Xte,recs,Cc,eps=cfg.EPSILON,iters=cfg.SINKHORN_PAIR_ITERS)
 def ot1(): ot1nn_predict(Xte,Xtr,ytr,Cc,eps=cfg.EPSILON,iters=cfg.SINKHORN_PAIR_ITERS)
 def taot(): taot_scores(Xte,Xtr,ytr,eps=cfg.EPSILON,lam=cfg.LAMBDA,spatial_weight=cfg.ETA,circular=True,steps=cfg.ADAM_STEPS,lr=cfg.ADAM_LR,opt_sinkhorn_iters=cfg.SINKHORN_OPT_ITERS,final_sinkhorn_iters=cfg.SINKHORN_FINAL_ITERS)
 def svm(): SVC(C=cfg.SVM_C,gamma=cfg.SVM_GAMMA).fit(Xtr,ytr).predict(Xte)
 rows=[]
 for name,fn in [('Signed-L1-SRC',src),('CircularOT-SRC',circ_post),('CircularOT-1NN',ot1),('TAOT-SRC',taot),('RBF-SVM train+predict',svm)]:
  arr=timed(fn,args.repeats); rows.append({'method':name,'repeats':args.repeats,'mean_total_s':arr.mean(),'std_total_s':arr.std(ddof=1) if len(arr)>1 else 0.0,'ms_per_query':1000*arr.mean()/len(Xte)})
 pd.DataFrame(rows).to_csv(out/'runtime.csv',index=False)
 meta={'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'repeats':args.repeats,'nqueries':len(Xte),'feature_extraction_excluded':True}
 (root/'provenance'/'runtime_environment.json').write_text(json.dumps(meta,indent=2))
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
