"""Classical baselines and NumPy OT utilities."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def src_scores(Xte, Xtr, ytr, alpha=1e-4, positive=False):
    D=np.asarray(Xtr,float).T; ytr=np.asarray(ytr); classes=np.unique(ytr)
    out=[]; recs=[]
    for y in np.asarray(Xte,float):
        coef=Lasso(alpha=alpha,positive=positive,fit_intercept=False,
                   max_iter=10000,tol=1e-5).fit(D,y).coef_
        s=[]; rr=[]
        for c in classes:
            cc=coef.copy(); cc[ytr != c]=0
            rec=D@cc; rr.append(rec); s.append(np.linalg.norm(y-rec))
        out.append(s); recs.append(rr)
    return np.asarray(out),np.asarray(recs),classes


def crc_scores(Xte,Xtr,ytr,ridge=1e-2):
    D=np.asarray(Xtr,float).T; Xte=np.asarray(Xte,float); ytr=np.asarray(ytr)
    classes=np.unique(ytr)
    A=np.linalg.solve(D.T@D+ridge*np.eye(D.shape[1]),D.T@Xte.T)
    out=[]
    for j,y in enumerate(Xte):
        s=[]
        for c in classes:
            cc=A[:,j].copy(); cc[ytr != c]=0
            s.append(np.linalg.norm(y-D@cc))
        out.append(s)
    return np.asarray(out),classes


def sinkhorn_np_pairs(A,B,C,eps=0.01,iters=60):
    A=np.asarray(A,float); B=np.asarray(B,float); C=np.asarray(C,float)
    K=np.exp(-C/eps); KC=K*C; out=[]
    for a,b in zip(A,B):
        a=np.maximum(a,1e-12); a=a/a.sum()
        b=np.maximum(b,1e-12); b=b/b.sum()
        u=np.ones_like(a); v=np.ones_like(b)
        for _ in range(iters):
            u=a/(K@v+1e-15); v=b/(K.T@u+1e-15)
        out.append(np.sum((u[:,None]*K*v[None,:])*C))
    return np.asarray(out)


def posthoc_ot_scores(Xte,recs,C,eps=0.01,iters=60):
    n,c,d=recs.shape
    A=np.repeat(Xte,c,axis=0); B=recs.reshape(n*c,d)
    return sinkhorn_np_pairs(A,B,C,eps=eps,iters=iters).reshape(n,c)


def ot1nn_predict(Xte,Xtr,ytr,C,eps=0.01,iters=60):
    n,m=len(Xte),len(Xtr)
    A=np.repeat(Xte,m,axis=0); B=np.tile(Xtr,(n,1))
    s=sinkhorn_np_pairs(A,B,C,eps=eps,iters=iters).reshape(n,m)
    return np.asarray(ytr)[np.argmin(s,axis=1)]


def svm_accuracy(Xtr,ytr,Xte,yte,C=3.0,gamma='scale'):
    svm=SVC(C=C,gamma=gamma).fit(Xtr,ytr)
    return float(svm.score(Xte,yte))
