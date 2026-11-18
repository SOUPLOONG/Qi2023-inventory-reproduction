import os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.name=='nt':sys.path.insert(0,str(ROOT/'.vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/test_matplotlib'))
import numpy as np
import torch
from scipy.optimize import linprog
from inventory_full.core import label,advance
from inventory_full.data import generate,features,build_samples,labels_for
from inventory_full.engine import plan,simulate
from inventory_full.models import InventoryNet
from inventory_full.training import predict

class InventoryTests(unittest.TestCase):
    def test_oracle_against_independent_global_lp_120_cases(self):
        rng=np.random.default_rng(981)
        for case in range(120):
            T=int(rng.integers(15,31));R=int(rng.integers(5,11));M=(T-1)//R+1
            leads=rng.integers(1,5,M+1);arrival=np.arange(M+1)*R+leads
            demand=rng.integers(0,20,T).astype(float);initial=float(rng.integers(-15,30))
            b=float(rng.choice([1,2,9,99]));stock=initial;observed=0.;queue={}
            for day in range(T):
                stock+=queue.pop(day,0.)
                if day%R==0:
                    m=day//R;end=min(int(arrival[m+1]),T)
                    q=label(demand[day:end],stock,int(leads[m]),end-day,b,1.) if day+leads[m]<end else 0.
                    queue[day+int(leads[m])]=q
                stock-=demand[day];observed+=max(stock,0)+b*max(-stock,0)
            obj=np.r_[np.zeros(M),np.ones(T),np.full(T,b)]
            A=np.zeros((T,M+2*T))
            for day in range(T):
                A[day,:M]=-(arrival[:M]<=day).astype(float);A[day,M+day]=1;A[day,M+T+day]=-1
            sol=linprog(obj,A_eq=A,b_eq=initial-demand.cumsum(),bounds=(0,None),method='highs')
            self.assertTrue(sol.success);self.assertAlmostEqual(observed,sol.fun,places=7,msg=str(case))

    def test_piecewise_cost_and_hybrid_lost_once(self):
        inv,H,B,lost=advance(np.array([3.,-2.]),np.array([0.,1.]),np.array([8.,4.]),np.array([2.,0.]),h=1,b=9,mode='hybrid')
        np.testing.assert_equal(inv,[-2.,0.]);np.testing.assert_equal(lost,[3.,5.])
        np.testing.assert_equal(B,[45.,45.]);np.testing.assert_equal(H,[0.,0.])
        inv2,_,B2,lost2=advance(inv,np.zeros(2),np.zeros(2),np.array([2.,0.]),h=1,b=9,mode='hybrid')
        np.testing.assert_equal(lost2,[0.,0.]);np.testing.assert_equal(B2,[18.,0.])

    def test_pipeline_causality_and_model_serialization(self):
        cfg={'seed':178,'n_series':8,'days':420,'horizon':20,'device':'cpu'}
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            p=generate(Path(td),cfg);ds=build_samples(p,cfg,Path(td))
            self.assertLessEqual(int((ds['day'][ds['split']=='train']+19).max()),279)
            self.assertLessEqual(int((ds['day'][ds['split']=='val']+19).max()),329)
            ids=np.arange(8);times=np.full(8,330);stock=np.ones(8)
            x=features(p,ids,times,stock);p2={k:v.copy() for k,v in p.items()}
            p2['demand'][:,330:]+=123;p2['lead'][:,330:]=4
            x2=features(p2,ids,times,stock)
            for k in x:np.testing.assert_equal(x[k],x2[k],err_msg=k)
            self.assertEqual(84+x['static'].shape[1]+x['vl'].shape[1]+x['stats'].shape[1]+2,132)
            np.testing.assert_allclose(labels_for(ds,9),ds['label'])
            model=InventoryNet();model.eval();a=predict(model,x,cfg)
            path=Path(td)/'weights.pt';torch.save(model.state_dict(),path)
            other=InventoryNet();other.load_state_dict(torch.load(path,weights_only=True));other.eval()
            aa=predict(other,x,cfg)
            for k in a:np.testing.assert_equal(a[k],aa[k])
            self.assertTrue((np.diff(a['daily'],axis=-1)>=0).all())
            # Earlier forks may not read later demand observations.
            tx={k:torch.as_tensor(v,dtype=torch.float32) for k,v in x.items()}
            changed={k:v.clone() for k,v in tx.items()};changed['seq'][:,20:]+=500
            with torch.no_grad():
                early=model(tx)['daily'][:,0];early2=model(changed)['daily'][:,0]
            torch.testing.assert_close(early,early2,rtol=0,atol=0)

    def test_continuous_field_boundary_equivalence(self):
        cfg={'seed':531,'n_series':8,'days':420,'horizon':20,'device':'cpu'}
        with tempfile.TemporaryDirectory() as td:
            p=generate(Path(td),cfg)
            P=plan(p,300,100,cfg,field=True,anchor=300)
            whole,tr,od,iv=simulate(p,P,cfg,300,100,'Normal',h=12,b=88,field=True,keep=True)
            state=None;ivs=[];cost=np.zeros(8);orders=[]
            for start,length in [(300,30),(330,30),(360,10),(370,30)]:
                nxt={};P=plan(p,start,length,cfg,field=True,anchor=300)
                df,_,o,i=simulate(p,P,cfg,start,length,'Normal',h=12,b=88,field=True,keep=True,initial_state=state,final_state=nxt)
                cost+=df.total_cost.to_numpy();ivs.append(i);orders.extend(o.order_quantity);state=nxt
            np.testing.assert_allclose(cost,whole.total_cost,rtol=1e-12)
            np.testing.assert_equal(np.concatenate(ivs,axis=1),iv)
            np.testing.assert_equal(np.array(orders),od.order_quantity.to_numpy())

if __name__=='__main__':unittest.main()
