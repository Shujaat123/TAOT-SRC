#!/usr/bin/env python3
"""Recompute summaries and paired statistical tests from per-seed outputs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

METHODS=['SRC-L2','NN-L1-SRC','CRC-L2','LinearOT-SRC','CircularOT-SRC','CircularOT-1NN','RBF-SVM','GS-SRC-linear','GS-SRC-circular']

def fmt_summary(d,group,dataset):
 rows=[]
 for key,g in d.groupby(group):
  for m in METHODS:
   if m in g:
    rows.append({'dataset':dataset,group:key,'method':m,'mean':g[m].mean(),'std':g[m].std(ddof=1)})
  if 'GS-effective-atoms' in g:
    rows.append({'dataset':dataset,group:key,'method':'GS-effective-atoms','mean':g['GS-effective-atoms'].mean(),'std':g['GS-effective-atoms'].std(ddof=1)})
 return rows

def paired_stats(d,group,dataset):
 rows=[]
 for key,g in d.groupby(group):
  a=g['SRC-L2'].to_numpy(); b=g['GS-SRC-circular'].to_numpy(); diff=b-a
  t,p=ttest_rel(b,a)
  try:
   w,wp=wilcoxon(b,a,alternative='two-sided',zero_method='wilcox')
  except Exception:
   w,wp=np.nan,np.nan
  rows.append({'dataset':dataset,group:key,'n':len(g),'src_mean':a.mean(),'taot_mean':b.mean(),'delta':diff.mean(),'t_stat':t,'t_pvalue':p,'wilcoxon_stat':w,'wilcoxon_pvalue':wp})
 return rows

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='results/reference'); ap.add_argument('--outdir',default='results/derived'); ap.add_argument('--root',default=str(Path(__file__).resolve().parent.parent)); args=ap.parse_args()
 root=Path(args.root).resolve(); inp=Path(args.input_dir); inp=inp if inp.is_absolute() else root/inp; out=Path(args.outdir); out=out if out.is_absolute() else root/out; out.mkdir(parents=True,exist_ok=True)
 rot_sum=[]; rot_stats=[]; few_sum=[]; few_stats=[]
 for ds in ['Digits','LFW-Face']:
  d=pd.read_csv(inp/f'heldout_rotation_{ds}.csv'); rot_sum+=fmt_summary(d,'rotation',ds); rot_stats+=paired_stats(d,'rotation',ds)
  d=pd.read_csv(inp/f'heldout_fewshot_{ds}.csv'); few_sum+=fmt_summary(d,'k',ds); few_stats+=paired_stats(d,'k',ds)
 pd.DataFrame(rot_sum).to_csv(out/'rotation_summary.csv',index=False); pd.DataFrame(rot_stats).to_csv(out/'rotation_stats.csv',index=False)
 pd.DataFrame(few_sum).to_csv(out/'fewshot_summary.csv',index=False); pd.DataFrame(few_stats).to_csv(out/'fewshot_stats.csv',index=False)
 s=pd.read_csv(inp/'synthetic_hairpin.csv'); sm=[]
 for m in [c for c in s.columns if c!='seed']:
  sm.append({'method':m,'mean':s[m].mean(),'std':s[m].std(ddof=1)})
 pd.DataFrame(sm).to_csv(out/'synthetic_summary.csv',index=False)
 t,p=ttest_rel(s['GS-SRC-Geodesic'],s['SRC-L2']);
 try: w,wp=wilcoxon(s['GS-SRC-Geodesic'],s['SRC-L2'])
 except Exception: w,wp=np.nan,np.nan
 pd.DataFrame([{'comparison':'GS-SRC-Geodesic vs SRC-L2','n':len(s),'delta':(s['GS-SRC-Geodesic']-s['SRC-L2']).mean(),'t_stat':t,'t_pvalue':p,'wilcoxon_stat':w,'wilcoxon_pvalue':wp}]).to_csv(out/'synthetic_stats.csv',index=False)
 print('Wrote derived summaries/statistics to',out)
if __name__=='__main__': main()
