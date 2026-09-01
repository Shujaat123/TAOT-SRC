#!/usr/bin/env python3
"""Create exact split/rotation manifests and feature caches from public data."""
from __future__ import annotations
import argparse, hashlib, json, sys
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
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()


def datasets():
    ds=load_digits()
    digits=(ds.images.astype(float), ds.target.astype(int))
    lfw=data.lfw_subset().astype(float)
    lfw_y=np.r_[np.zeros(100,dtype=int), np.ones(100,dtype=int)]
    return {'Digits':digits, 'LFW-Face':(lfw,lfw_y)}


def build_manifest(y, seed, k, ntest, rotation, max_k=None):
    """Build a paired split/rotation manifest.

    Protocol v4:
    * one deterministic class-wise permutation per seed;
    * first ``max_k`` samples form a nested training pool;
    * the next ``ntest`` samples form a test set shared by every shot count;
    * rotation factors are drawn by an RNG independent of the split RNG;
    * the same factor is scaled by 10 or 20 degrees, yielding paired severity.
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

    # Independent perturbation RNG.  The same unit factors are reused across
    # rotation severities so that +/−20° is a scaled version of +/−10°.
    angle_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 424242]))
    factors = angle_rng.uniform(-1.0, 1.0, size=len(test_idx))
    test_angles = (factors * float(rotation)).astype(float).tolist() if rotation else [0.0] * len(test_idx)

    return dict(seed=int(seed), k=int(k), max_k=int(max_k), ntest=int(ntest),
                rotation=float(rotation), train_idx=train_idx, test_idx=test_idx,
                test_angles=test_angles, ytrain=ytr, ytest=yte)


def feature_from_manifest(images, man):
    tr=images[np.asarray(man['train_idx'],int)]
    te=[]
    for ii,ang in zip(man['test_idx'],man['test_angles']):
        im=images[int(ii)]
        if ang:
            im=ndi_rotate(im,ang,reshape=False,order=1,mode='constant',cval=0)
        te.append(im)
    Xtr=np.asarray([hog_hist(im,cells=cfg.CELLS,nbins=cfg.NBINS,floor=cfg.HOG_FLOOR) for im in tr])
    Xte=np.asarray([hog_hist(im,cells=cfg.CELLS,nbins=cfg.NBINS,floor=cfg.HOG_FLOOR) for im in te])
    return Xtr,np.asarray(man['ytrain'],int),Xte,np.asarray(man['ytest'],int)


def gaussian_hist(center_idx,n=40,sigma=1.8):
    idx=np.arange(n); h=np.exp(-0.5*((idx-center_idx)/sigma)**2); return h/h.sum()


def synthetic_split(seed,ntrain=4,ntest=50,noise=0.02):
    rng=np.random.default_rng(seed); centers=[8,31,20]; Xtr=[];ytr=[];Xte=[];yte=[]
    for c,mu in enumerate(centers):
        for _ in range(ntrain):
            h=gaussian_hist(mu+rng.normal(0,1)); h=np.maximum(h+rng.normal(0,noise,40),0); h/=h.sum(); Xtr.append(h); ytr.append(c)
        for _ in range(ntest):
            ctr=np.clip(mu+rng.normal(0,6),0,39); h=gaussian_hist(ctr); h=np.maximum(h+rng.normal(0,noise,40),0); h/=h.sum(); Xte.append(h); yte.append(c)
    return np.asarray(Xtr),np.asarray(ytr),np.asarray(Xte),np.asarray(yte)


def hairpin_costs(n_arm=18,n_conn=4,gap=0.12):
    x1=np.linspace(0,1,n_arm); y1=np.zeros(n_arm)
    yc=np.linspace(0,gap,n_conn+2)[1:-1]; xc=np.ones(n_conn)
    x2=np.linspace(1,0,n_arm); y2=np.full(n_arm,gap)
    coords=np.c_[np.r_[x1,xc,x2],np.r_[y1,yc,y2]]
    ds=np.sqrt(((coords[1:]-coords[:-1])**2).sum(1)); s=np.r_[0,np.cumsum(ds)]
    Ce=((coords[:,None,:]-coords[None,:,:])**2).sum(-1)
    Cg=(s[:,None]-s[None,:])**2
    return Ce/Ce.max(),Cg/Cg.max()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(HERE.parent)); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args()
    root=Path(args.root).resolve(); mdir=root/'data'/'manifests'; fdir=root/'data'/'features'; pdir=root/'provenance'
    mdir.mkdir(parents=True,exist_ok=True); fdir.mkdir(parents=True,exist_ok=True); pdir.mkdir(parents=True,exist_ok=True)
    dsets=datasets(); records=[]
    jobs=[]
    for ds in cfg.DATASETS:
        ntest=cfg.DIGITS_NTEST if ds=='Digits' else cfg.LFW_NTEST
        for seed in cfg.HELDOUT_SEEDS:
            for rot in cfg.ROTATIONS: jobs.append((ds,seed,cfg.DEFAULT_SHOT,ntest,rot,'rotation'))
            for k in cfg.FEWSHOT: jobs.append((ds,seed,k,ntest,0,'fewshot'))
        # validation cache covers parameter sweeps
        for seed in cfg.VALIDATION_SEEDS:
            for rot in cfg.ROTATIONS: jobs.append((ds,seed,cfg.DEFAULT_SHOT,ntest,rot,'validation'))
    seen=set()
    for ds,seed,k,ntest,rot,purpose in jobs:
        key=(ds,seed,k,rot)
        # identical feature condition may serve multiple purposes
        if key in seen: continue
        seen.add(key)
        images,y=dsets[ds]
        man=build_manifest(y,seed,k,ntest,rot)
        stem=f"{ds.replace('-','_')}_seed{seed}_k{k}_rot{int(rot)}"
        mp=mdir/f'{stem}.json'; fp=fdir/f'{stem}.npz'
        mp.write_text(json.dumps({'dataset':ds,'purpose':purpose,**man},indent=2))
        Xtr,ytr,Xte,yte=feature_from_manifest(images,man)
        np.savez_compressed(fp,Xtrain=Xtr,ytrain=ytr,Xtest=Xte,ytest=yte)
        records.append({'dataset':ds,'seed':seed,'k':k,'rotation':rot,'manifest':mp.relative_to(root).as_posix(),'feature':fp.relative_to(root).as_posix(),'manifest_sha256':sha256(mp),'feature_sha256':sha256(fp),'ntrain':len(ytr),'ntest':len(yte)})
    Ce,Cg=hairpin_costs(); np.savez_compressed(fdir/'synthetic_costs.npz',C_euclidean=Ce,C_geodesic=Cg)
    for seed in range(20):
        Xtr,ytr,Xte,yte=synthetic_split(seed)
        fp=fdir/f'synthetic_seed{seed}.npz'; np.savez_compressed(fp,Xtrain=Xtr,ytrain=ytr,Xtest=Xte,ytest=yte)
        records.append({'dataset':'Synthetic-Hairpin','seed':seed,'k':4,'rotation':0,'feature':fp.relative_to(root).as_posix(),'feature_sha256':sha256(fp),'ntrain':len(ytr),'ntest':len(yte)})
    meta={'python':sys.version,'config':{k:getattr(cfg,k) for k in ['CELLS','NBINS','HOG_FLOOR','ETA','EPSILON','LAMBDA','ADAM_STEPS','ADAM_LR','SINKHORN_OPT_ITERS','SINKHORN_FINAL_ITERS','SINKHORN_PAIR_ITERS','SRC_ALPHA','CRC_RIDGE','SVM_C','VALIDATION_SEEDS','HELDOUT_SEEDS','ROTATIONS','FEWSHOT']},'records':records}
    (pdir/'reference_data_metadata.json').write_text(json.dumps(meta,indent=2,default=list))
    print(f'Prepared {len(records)} cached conditions under {root}')

if __name__=='__main__': main()
