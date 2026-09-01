#!/usr/bin/env python3
"""Full held-out, few-shot, and synthetic experiment driver using exact caches."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import config as cfg
from taot_src import ground_cost, taot_scores, sinkhorn_transport_cost_batch, coefficient_entropy
from experiment_utils import src_scores,crc_scores,posthoc_ot_scores,ot1nn_predict,svm_accuracy

# Deterministic CPU reference mode
torch.set_num_threads(1)
try: torch.use_deterministic_algorithms(True)
except Exception: pass


def stem(dataset,seed,k,rotation):
    return f"{dataset.replace('-','_')}_seed{seed}_k{k}_rot{int(rotation)}.npz"


def load_condition(root,dataset,seed,k,rotation):
    p=Path(root)/'data'/'features'/stem(dataset,seed,k,rotation)
    z=np.load(p)
    return z['Xtrain'],z['ytrain'],z['Xtest'],z['ytest']


def evaluate_features(Xtr,ytr,Xte,yte, *, eps=cfg.EPSILON, lam=cfg.LAMBDA, eta=cfg.ETA):
    src,_,classes=src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=False)
    nn_src,recs,_=src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=True)
    crc,_=crc_scores(Xte,Xtr,ytr,ridge=cfg.CRC_RIDGE)
    Clinear=ground_cost(spatial_weight=eta,circular=False)
    Ccirc=ground_cost(spatial_weight=eta,circular=True)
    linear_post=posthoc_ot_scores(Xte,recs,Clinear,eps=eps,iters=cfg.SINKHORN_PAIR_ITERS)
    circ_post=posthoc_ot_scores(Xte,recs,Ccirc,eps=eps,iters=cfg.SINKHORN_PAIR_ITERS)
    nn_pred=ot1nn_predict(Xte,Xtr,ytr,Ccirc,eps=eps,iters=cfg.SINKHORN_PAIR_ITERS)
    sl,cl,El=taot_scores(Xte,Xtr,ytr,eps=eps,lam=lam,spatial_weight=eta,circular=False,
                         steps=cfg.ADAM_STEPS,lr=cfg.ADAM_LR,opt_sinkhorn_iters=cfg.SINKHORN_OPT_ITERS,
                         final_sinkhorn_iters=cfg.SINKHORN_FINAL_ITERS,return_effective_atoms=True)
    sg,cg,Eg=taot_scores(Xte,Xtr,ytr,eps=eps,lam=lam,spatial_weight=eta,circular=True,
                         steps=cfg.ADAM_STEPS,lr=cfg.ADAM_LR,opt_sinkhorn_iters=cfg.SINKHORN_OPT_ITERS,
                         final_sinkhorn_iters=cfg.SINKHORN_FINAL_ITERS,return_effective_atoms=True)
    predg=cg[sg.argmin(1)]; win_eff=Eg[np.arange(len(yte)),sg.argmin(1)]
    return {
      'SRC-L2':accuracy_score(yte,classes[src.argmin(1)]),
      'NN-L1-SRC':accuracy_score(yte,classes[nn_src.argmin(1)]),
      'CRC-L2':accuracy_score(yte,classes[crc.argmin(1)]),
      'LinearOT-SRC':accuracy_score(yte,classes[linear_post.argmin(1)]),
      'CircularOT-SRC':accuracy_score(yte,classes[circ_post.argmin(1)]),
      'CircularOT-1NN':accuracy_score(yte,nn_pred),
      'RBF-SVM':svm_accuracy(Xtr,ytr,Xte,yte,C=cfg.SVM_C,gamma=cfg.SVM_GAMMA),
      'GS-SRC-linear':accuracy_score(yte,cl[sl.argmin(1)]),
      'GS-SRC-circular':accuracy_score(yte,predg),
      'GS-effective-atoms':float(np.mean(win_eff)),
    }


def taot_arbitrary_cost(Xte,Xtr,ytr,C,eps=cfg.SYNTH_EPSILON,lam=cfg.SYNTH_LAMBDA):
    A=torch.tensor(Xte,dtype=torch.float32); Ct=torch.tensor(C,dtype=torch.float32); scores=[]
    for c in np.unique(ytr):
        Dc=torch.tensor(Xtr[ytr==c],dtype=torch.float32)
        th=torch.zeros((len(Xte),len(Dc)),dtype=torch.float32,requires_grad=True)
        opt=torch.optim.Adam([th],lr=cfg.ADAM_LR)
        for _ in range(cfg.ADAM_STEPS):
            opt.zero_grad(); w=torch.softmax(th,1); rec=w@Dc
            ot=sinkhorn_transport_cost_batch(A,rec,Ct,eps=eps,n_iters=cfg.SINKHORN_OPT_ITERS)
            H=coefficient_entropy(w); loss=(ot+lam*H).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            w=torch.softmax(th,1); rec=w@Dc
            ot=sinkhorn_transport_cost_batch(A,rec,Ct,eps=eps,n_iters=cfg.SINKHORN_FINAL_ITERS)
            H=coefficient_entropy(w); scores.append((ot+lam*H).cpu().numpy())
    return np.stack(scores,1)


def run_synthetic(root,outdir,seeds):
    costs=np.load(Path(root)/'data'/'features'/'synthetic_costs.npz'); Ce=costs['C_euclidean']; Cg=costs['C_geodesic']
    rows=[]
    for seed in seeds:
        z=np.load(Path(root)/'data'/'features'/f'synthetic_seed{seed}.npz')
        Xtr,ytr,Xte,yte=z['Xtrain'],z['ytrain'],z['Xtest'],z['ytest']
        src,_,classes=src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=False)
        nn,_,_=src_scores(Xte,Xtr,ytr,alpha=cfg.SRC_ALPHA,positive=True)
        se=taot_arbitrary_cost(Xte,Xtr,ytr,Ce); sg=taot_arbitrary_cost(Xte,Xtr,ytr,Cg)
        rows.append({'seed':seed,'SRC-L2':accuracy_score(yte,classes[src.argmin(1)]),
                     'NN-L1-SRC':accuracy_score(yte,classes[nn.argmin(1)]),
                     'GS-SRC-Euclidean':accuracy_score(yte,se.argmin(1)),
                     'GS-SRC-Geodesic':accuracy_score(yte,sg.argmin(1))})
    pd.DataFrame(rows).to_csv(Path(outdir)/'synthetic_hairpin.csv',index=False)


def run_real(root,outdir,dataset,seeds,rotations,shots, *, eps=cfg.EPSILON, lam=cfg.LAMBDA, eta=cfg.ETA):
    rotrows=[]
    for r in rotations:
        for seed in seeds:
            Xtr,ytr,Xte,yte=load_condition(root,dataset,seed,cfg.DEFAULT_SHOT,r)
            row=evaluate_features(Xtr,ytr,Xte,yte,eps=eps,lam=lam,eta=eta)
            # Effective atom count is a few-shot concentration diagnostic. It is
            # intentionally not part of the rotation result record because it is
            # not reported or interpreted for the rotation experiment.
            row.pop('GS-effective-atoms', None)
            row.update(dataset=dataset,rotation=r,seed=seed); rotrows.append(row)
    pd.DataFrame(rotrows).to_csv(Path(outdir)/f'heldout_rotation_{dataset}.csv',index=False)
    fs=[]
    for k in shots:
        for seed in seeds:
            Xtr,ytr,Xte,yte=load_condition(root,dataset,seed,k,0)
            row=evaluate_features(Xtr,ytr,Xte,yte,eps=eps,lam=lam,eta=eta); row.update(dataset=dataset,k=k,seed=seed); fs.append(row)
    pd.DataFrame(fs).to_csv(Path(outdir)/f'heldout_fewshot_{dataset}.csv',index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(HERE.parent)); ap.add_argument('--outdir',default='results/rerun_full')
    ap.add_argument('--quick',action='store_true'); ap.add_argument('--datasets',nargs='*',default=list(cfg.DATASETS))
    ap.add_argument('--eta',type=float,default=cfg.ETA)
    ap.add_argument('--epsilon',type=float,default=cfg.EPSILON)
    ap.add_argument('--lambda',dest='lam',type=float,default=cfg.LAMBDA,help='TAOT entropy weight')
    ap.add_argument('--selected-config',default=None,help='Optional JSON produced by run_validation.py')
    args=ap.parse_args()
    root=Path(args.root).resolve(); out=Path(args.outdir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
    eta,eps,lam=args.eta,args.epsilon,args.lam
    if args.selected_config:
        import json
        hp=json.loads(Path(args.selected_config).read_text())
        eta=float(hp['eta']); eps=float(hp['epsilon']); lam=float(hp['lambda'])
    print(f'Using real-data hyperparameters: eta={eta}, epsilon={eps}, lambda={lam}')
    seeds=[cfg.HELDOUT_SEEDS[0]] if args.quick else list(cfg.HELDOUT_SEEDS)
    rotations=(0,) if args.quick else cfg.ROTATIONS; shots=(cfg.DEFAULT_SHOT,) if args.quick else cfg.FEWSHOT
    for ds in args.datasets: run_real(root,out,ds,seeds,rotations,shots,eps=eps,lam=lam,eta=eta)
    run_synthetic(root,out,[0] if args.quick else list(cfg.SYNTH_SEEDS))
    print('Wrote results to',out)

if __name__=='__main__': main()
