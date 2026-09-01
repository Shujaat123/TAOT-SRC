#!/usr/bin/env python3
"""Core TAOT-SRC implementation used by the exact reproducibility package.

Important: the score used by the experiments is <P_epsilon, C>, the
transport term of the entropically regularized Sinkhorn coupling.  It is
not a debiased Sinkhorn divergence and not the full regularized primal value.
"""
from __future__ import annotations
import numpy as np
import torch


def hog_hist(image, cells=(2, 2), nbins=9, floor=1e-6):
    image = np.asarray(image, dtype=float)
    gy, gx = np.gradient(image)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    H, W = image.shape
    feats = []
    for iy in range(cells[0]):
        y0, y1 = iy * H // cells[0], (iy + 1) * H // cells[0]
        for ix in range(cells[1]):
            x0, x1 = ix * W // cells[1], (ix + 1) * W // cells[1]
            a = ang[y0:y1, x0:x1].ravel()
            m = mag[y0:y1, x0:x1].ravel()
            bins = np.floor(a / np.pi * nbins).astype(int) % nbins
            feats.append(np.bincount(bins, weights=m, minlength=nbins))
    h = np.concatenate(feats).astype(float) + floor
    return h / h.sum()


def ground_cost(cells=(2, 2), nbins=9, spatial_weight=0.5, circular=True):
    pts = []
    for iy in range(cells[0]):
        for ix in range(cells[1]):
            for b in range(nbins):
                theta = (b + 0.5) * np.pi / nbins
                pts.append((iy, ix, theta))
    C = np.zeros((len(pts), len(pts)), dtype=float)
    scale = max(cells) ** 2
    for i, (y1, x1, t1) in enumerate(pts):
        for j, (y2, x2, t2) in enumerate(pts):
            ds2 = ((y1-y2)**2 + (x1-x2)**2) / scale
            dt = abs(t1-t2)
            if circular:
                dt = min(dt, np.pi-dt)
            da2 = (dt/(np.pi/2))**2
            C[i, j] = spatial_weight*ds2 + da2
    return C / (C.max() + 1e-12)


def sinkhorn_plan_batch(a, b, C, eps=0.01, n_iters=30):
    """Return the entropically regularized Sinkhorn coupling for batched marginals."""
    tiny = 1e-9
    a = a.clamp_min(tiny); a = a / a.sum(1, keepdim=True)
    b = b.clamp_min(tiny); b = b / b.sum(1, keepdim=True)
    K = torch.exp(-C / eps).clamp_min(1e-30)
    u = torch.ones_like(a) / a.shape[1]
    v = torch.ones_like(b) / b.shape[1]
    for _ in range(n_iters):
        u = a / (v @ K.T + tiny)
        v = b / (u @ K + tiny)
    return u[:, :, None] * K[None, :, :] * v[:, None, :]


def sinkhorn_transport_cost_batch(a, b, C, eps=0.01, n_iters=30):
    P = sinkhorn_plan_batch(a, b, C, eps=eps, n_iters=n_iters)
    return (P * C[None, :, :]).sum((1, 2))


def sinkhorn_full_primal_batch(a, b, C, eps=0.01, n_iters=30):
    """Diagnostic only: full regularized primal value, not the reference score."""
    P = sinkhorn_plan_batch(a, b, C, eps=eps, n_iters=n_iters)
    transport = (P * C[None, :, :]).sum((1, 2))
    entropy_term = eps * (P * (P.clamp_min(1e-30).log() - 1.0)).sum((1, 2))
    return transport + entropy_term

# Backward-compatible name used by earlier scripts.
sinkhorn_cost_batch = sinkhorn_transport_cost_batch


def coefficient_entropy(w, delta=1e-9):
    """Entropy penalty used by TAOT-SRC: -sum_j w_j log(w_j + delta)."""
    return -(w * torch.log(w + delta)).sum(1)


def taot_scores(Xtest, Xtrain, ytrain, *, eps=0.01, lam=0.0,
                spatial_weight=0.5, circular=True, steps=70, lr=0.35,
                opt_sinkhorn_iters=30, final_sinkhorn_iters=50,
                return_effective_atoms=False):
    Xtest = np.asarray(Xtest, float)
    Xtrain = np.asarray(Xtrain, float)
    ytrain = np.asarray(ytrain)
    classes = np.unique(ytrain)
    C = torch.tensor(ground_cost(spatial_weight=spatial_weight,
                                 circular=circular), dtype=torch.float32)
    A = torch.tensor(Xtest, dtype=torch.float32)
    scores, effs = [], []
    for c in classes:
        Dc = torch.tensor(Xtrain[ytrain == c], dtype=torch.float32)
        theta = torch.zeros((len(Xtest), len(Dc)), dtype=torch.float32,
                            requires_grad=True)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            w = torch.softmax(theta, 1)
            rec = w @ Dc
            ot = sinkhorn_transport_cost_batch(A, rec, C, eps=eps,
                                                n_iters=opt_sinkhorn_iters)
            entropy = coefficient_entropy(w)
            loss = (ot + lam * entropy).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            w = torch.softmax(theta, 1)
            rec = w @ Dc
            ot = sinkhorn_transport_cost_batch(A, rec, C, eps=eps,
                                                n_iters=final_sinkhorn_iters)
            entropy = coefficient_entropy(w)
            scores.append((ot + lam * entropy).cpu().numpy())
            effs.append(torch.exp(entropy).cpu().numpy())
    S = np.stack(scores, axis=1)
    E = np.stack(effs, axis=1)
    if return_effective_atoms:
        return S, classes, E
    return S, classes


def predict(Xtest, Xtrain, ytrain, **kwargs):
    scores, classes = taot_scores(Xtest, Xtrain, ytrain, **kwargs)
    return classes[np.argmin(scores, axis=1)]
