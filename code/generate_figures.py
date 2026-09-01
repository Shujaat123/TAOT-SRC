#!/usr/bin/env python3
"""Generate result visualizations from the authoritative reference CSVs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save(fig,path): fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
def rotation_plot(inp,ds,path):
 d=pd.read_csv(inp/f'heldout_rotation_{ds}.csv'); fig,ax=plt.subplots(figsize=(4.4,3.0))
 for m,label in [('SRC-L2','SRC'),('CircularOT-1NN','OT-1NN'),('RBF-SVM','RBF-SVM'),('GS-SRC-circular','TAOT-SRC')]:
  g=d.groupby('rotation')[m]; x=np.array(sorted(d.rotation.unique())); mu=np.array([g.get_group(v).mean() for v in x]); sd=np.array([g.get_group(v).std(ddof=1) for v in x]); ax.errorbar(x,100*mu,yerr=100*sd,marker='o',label=label)
 ax.set_xlabel('Maximum rotation (degrees)'); ax.set_ylabel('Accuracy (%)'); ax.legend(frameon=False,fontsize=8); save(fig,path)
def fewshot_plot(inp,ds,path):
 d=pd.read_csv(inp/f'heldout_fewshot_{ds}.csv'); fig,ax=plt.subplots(figsize=(4.4,3.0))
 for m,label in [('SRC-L2','SRC'),('CircularOT-1NN','OT-1NN'),('RBF-SVM','RBF-SVM'),('GS-SRC-circular','TAOT-SRC')]:
  g=d.groupby('k')[m]; x=np.array(sorted(d.k.unique())); mu=np.array([g.get_group(v).mean() for v in x]); sd=np.array([g.get_group(v).std(ddof=1) for v in x]); ax.errorbar(x,100*mu,yerr=100*sd,marker='o',label=label)
 ax.set_xlabel('Training samples per class'); ax.set_ylabel('Accuracy (%)'); ax.set_xticks(x); ax.legend(frameon=False,fontsize=8); save(fig,path)
def synthetic_plot(inp,path):
 d=pd.read_csv(inp/'synthetic_hairpin.csv'); cols=['SRC-L2','GS-SRC-Euclidean','GS-SRC-Geodesic']; means=[100*d[c].mean() for c in cols]; std=[100*d[c].std(ddof=1) for c in cols]
 fig,ax=plt.subplots(figsize=(4.4,3.0)); x=np.arange(3); ax.bar(x,means,yerr=std,capsize=3); ax.set_xticks(x,['SRC','Euclidean OT','Geodesic OT']); ax.set_ylabel('Accuracy (%)'); save(fig,path)
def topology(path):
 theta=np.linspace(0,np.pi,256,endpoint=False); ref=.03*np.pi; linear=np.abs(theta-ref)/(np.pi/2); circ=np.minimum(np.abs(theta-ref),np.pi-np.abs(theta-ref))/(np.pi/2)
 fig,ax=plt.subplots(figsize=(4.4,2.7)); ax.plot(theta/np.pi,linear,label='unwrapped'); ax.plot(theta/np.pi,circ,label='circular geodesic'); ax.set_xlabel(r'Orientation / $\pi$'); ax.set_ylabel('Normalized angular distance'); ax.legend(frameon=False); save(fig,path)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(Path(__file__).resolve().parent.parent)); ap.add_argument('--results-dir',default='results/reference'); ap.add_argument('--outdir',default='figures'); args=ap.parse_args(); root=Path(args.root).resolve(); inp=Path(args.results_dir); inp=inp if inp.is_absolute() else root/inp; out=Path(args.outdir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
 rotation_plot(inp,'Digits',out/'rotation_digits.pdf'); rotation_plot(inp,'LFW-Face',out/'rotation_lfw_face.pdf'); fewshot_plot(inp,'Digits',out/'fewshot_digits.pdf'); synthetic_plot(inp,out/'synthetic_hairpin_performance.pdf'); topology(out/'orientation_topology.pdf'); print('Figures written to',out)
if __name__=='__main__': main()
