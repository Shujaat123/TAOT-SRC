import sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'code'))
from taot_src import ground_cost,sinkhorn_transport_cost_batch,coefficient_entropy

def test_ground_cost():
 c=ground_cost(spatial_weight=.5,circular=True); l=ground_cost(spatial_weight=.5,circular=False)
 assert c.shape==(36,36); assert np.allclose(c,c.T); assert np.isclose(c.max(),1); assert c[0,8] < l[0,8]
def test_sinkhorn_finite():
 a=torch.tensor([[.25,.25,.25,.25]],dtype=torch.float32); C=torch.tensor(np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])/3,dtype=torch.float32)
 v=sinkhorn_transport_cost_batch(a,a,C,eps=.05,n_iters=50); assert torch.isfinite(v).all()


def test_entropy_matches_definition():
    w=torch.tensor([[0.2,0.3,0.5]],dtype=torch.float32)
    delta=1e-9
    expected=-(w*torch.log(w+delta)).sum(1)
    assert torch.allclose(coefficient_entropy(w,delta=delta),expected,atol=0,rtol=0)

def test_paired_split_and_nested_training():
    import sys
    sys.path.insert(0,str(ROOT/'code'))
    from prepare_reference_data import build_manifest
    y=np.repeat(np.arange(3),30)
    m1=build_manifest(y,seed=7,k=1,ntest=5,rotation=0,max_k=10)
    m5=build_manifest(y,seed=7,k=5,ntest=5,rotation=10,max_k=10)
    m10=build_manifest(y,seed=7,k=10,ntest=5,rotation=20,max_k=10)
    assert m1['test_idx']==m5['test_idx']==m10['test_idx']
    # per-class training sets are nested because each manifest takes a prefix of the same pool
    assert set(m1['train_idx']).issubset(set(m5['train_idx']))
    assert set(m5['train_idx']).issubset(set(m10['train_idx']))

def test_paired_rotation_scaling_and_bounds():
    import sys
    sys.path.insert(0,str(ROOT/'code'))
    from prepare_reference_data import build_manifest
    y=np.repeat(np.arange(2),30)
    m10=build_manifest(y,seed=11,k=5,ntest=5,rotation=10,max_k=10)
    m20=build_manifest(y,seed=11,k=5,ntest=5,rotation=20,max_k=10)
    a10=np.asarray(m10['test_angles']); a20=np.asarray(m20['test_angles'])
    assert np.all(np.abs(a10)<=10+1e-12); assert np.all(np.abs(a20)<=20+1e-12)
    assert np.allclose(a20,2*a10)
