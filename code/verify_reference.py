#!/usr/bin/env python3
"""Integrity, protocol, and mathematical checks for TAOT-SRC v4."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import config as cfg
from taot_src import ground_cost,sinkhorn_transport_cost_batch,sinkhorn_full_primal_batch,coefficient_entropy

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(HERE.parent)); args=ap.parse_args()
    root=Path(args.root).resolve(); errors=[]
    meta_path=root/'provenance'/'reference_data_metadata.json'
    if not meta_path.exists():
        errors.append('missing provenance/reference_data_metadata.json; run prepare_reference_data.py')
        meta={'records':[]}
    else:
        meta=json.loads(meta_path.read_text())
    for r in meta.get('records',[]):
        for key,hkey in [('manifest','manifest_sha256'),('feature','feature_sha256')]:
            if key not in r: continue
            rel=str(r[key]).replace('\\', '/')
            p=root/Path(rel)
            if not p.exists(): errors.append(f'missing {p}')
            elif sha256(p)!=r[hkey]: errors.append(f'hash mismatch {p}')

    # Mathematical checks.
    Cc=ground_cost(spatial_weight=cfg.ETA,circular=True); Cl=ground_cost(spatial_weight=cfg.ETA,circular=False)
    if not np.allclose(Cc,Cc.T): errors.append('circular ground cost not symmetric')
    if not np.isclose(Cc.max(),1.0,atol=1e-9): errors.append('circular ground cost not normalized')
    if not Cc[0,8] < Cl[0,8]: errors.append('circular wrap-around check failed')
    a=torch.tensor([[.4,.3,.2,.1]],dtype=torch.float32)
    C=torch.tensor(np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])/3,dtype=torch.float32)
    tc=sinkhorn_transport_cost_batch(a,a,C,eps=.05,n_iters=100)
    fp=sinkhorn_full_primal_batch(a,a,C,eps=.05,n_iters=100)
    if not (torch.isfinite(tc).all() and torch.isfinite(fp).all()): errors.append('Sinkhorn finite-value check failed')
    w=torch.tensor([[.2,.3,.5]],dtype=torch.float32)
    if not torch.equal(coefficient_entropy(w),-(w*torch.log(w+1e-9)).sum(1)):
        errors.append('coefficient entropy definition mismatch')

    # Protocol checks from shipped manifests.
    mdir=root/'data'/'manifests'
    for ds in cfg.DATASETS:
        tag=ds.replace('-','_')
        for seed in tuple(cfg.VALIDATION_SEEDS)+tuple(cfg.HELDOUT_SEEDS):
            # rotation pairing at default shot, if all conditions exist
            mans=[]
            for rot in cfg.ROTATIONS:
                p=mdir/f'{tag}_seed{seed}_k{cfg.DEFAULT_SHOT}_rot{int(rot)}.json'
                if p.exists(): mans.append(json.loads(p.read_text()))
            if len(mans)==len(cfg.ROTATIONS):
                if not all(m['test_idx']==mans[0]['test_idx'] for m in mans[1:]):
                    errors.append(f'{ds} seed {seed}: test indices differ across rotations')
                a10=np.asarray(mans[1]['test_angles']); a20=np.asarray(mans[2]['test_angles'])
                if not np.allclose(a20,2*a10): errors.append(f'{ds} seed {seed}: rotations are not paired/scaled')
            # few-shot pairing/nesting for held-out seeds
            if seed in cfg.HELDOUT_SEEDS:
                fs=[]
                for k in cfg.FEWSHOT:
                    p=mdir/f'{tag}_seed{seed}_k{k}_rot0.json'
                    if p.exists(): fs.append(json.loads(p.read_text()))
                if len(fs)==len(cfg.FEWSHOT):
                    if not all(m['test_idx']==fs[0]['test_idx'] for m in fs[1:]):
                        errors.append(f'{ds} seed {seed}: test indices differ across shot counts')
                    for a1,a2 in zip(fs,fs[1:]):
                        if not set(a1['train_idx']).issubset(set(a2['train_idx'])):
                            errors.append(f'{ds} seed {seed}: training sets are not nested')

    if errors:
        print('\n'.join('FAIL '+e for e in errors)); raise SystemExit(1)
    print(f'PASS: {len(meta.get("records",[]))} cached-condition hashes verified')
    print('PASS: paired rotation protocol')
    print('PASS: fixed-test nested few-shot protocol')
    print('PASS: exact entropy definition')
    print('PASS: ground-cost and Sinkhorn unit checks')
if __name__=='__main__': main()
