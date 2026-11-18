import os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if os.name=='nt':sys.path.insert(0,str(ROOT/'.vendor'))
import numpy as np
import torch
from inventory_full.data import generate
from inventory_full.models import InventoryNet
from inventory_full.engine import plan,simulate
from inventory_full.field_calibration import (FieldCalibrator,warm_state,episode,tensor_episode,rollout_torch,
    callback,attach_context,panel_digest)

class FieldV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(613)
        cls.tmp=tempfile.TemporaryDirectory();cls.cfg={'seed':178,'n_series':6,'days':420,'horizon':20,'device':'cpu'}
        cls.p=generate(Path(cls.tmp.name),cls.cfg);cls.encoder=InventoryNet();cls.encoder.eval()
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_baseline_initialization_and_nonnegative_actions(self):
        torch.manual_seed(61);model=FieldCalibrator().double()
        c=torch.randn(8,149,dtype=torch.float64);stock=torch.arange(8,dtype=torch.float64)
        openq=stock*2;scale=torch.ones(8,dtype=torch.float64);target=stock+5
        torch.testing.assert_close(model(c,stock,openq,scale,target),torch.relu(target-stock-openq),rtol=0,atol=0)

    def test_independent_simulators_four_environments_and_rollout_gradient(self):
        p=self.p;cfg=self.cfg;torch.manual_seed(614)
        model=FieldCalibrator().double();torch.nn.init.normal_(model.net[-1].weight,0,.02)
        for lead in [1,2]:
            for mode in ['backorder','hybrid']:
                state=warm_state(p,cfg,300,70,300,lead,mode)
                ep=episode(p,cfg,self.encoder,370,30,300,lead,mode,state)
                te=tensor_episode(ep,'cpu',torch.float64)
                result=rollout_torch(model,te,record=True)
                P=attach_context(p,plan(p,370,30,cfg,anchor=300,lead_scale=lead),self.encoder,cfg)
                df,tr,orders,_=simulate(p,P,cfg,370,30,'E2E_calibrated',h=12,b=88,lead_scale=lead,
                    shortage_mode=mode,initial_state=state,policy=callback(model,cfg),keep=True)
                for key in ['holding_cost','stockout_cost','total_cost','lost_units']:
                    np.testing.assert_allclose(result[key].detach().numpy(),df[key],atol=1e-7,rtol=1e-10)
                ordermat=np.zeros((30,6));ordermat[orders.order_day.to_numpy()-370,orders.series_id]=orders.order_quantity
                np.testing.assert_allclose(result['actions'].detach(),ordermat,atol=1e-9)
                model.zero_grad();result['total_cost'].mean().backward()
                self.assertTrue(torch.isfinite(model.net[-1].weight.grad).all())
                self.assertGreater(float(model.net[-1].weight.grad.norm()),0)

    def test_future_outcomes_not_policy_inputs_and_first_action_invariance(self):
        p=self.p;cfg=self.cfg;state=warm_state(p,cfg,300,70,300,2,'hybrid')
        p2={k:v.copy() for k,v in p.items()};p2['demand'][:,370:]+=333;p2['lead'][:,370:]=4
        P=attach_context(p,plan(p,370,30,cfg,anchor=300,lead_scale=2),self.encoder,cfg)
        P2=attach_context(p2,plan(p2,370,30,cfg,anchor=300,lead_scale=2),self.encoder,cfg)
        rows=np.flatnonzero(P['times']==370);rows2=np.flatnonzero(P2['times']==370)
        self.assertGreater(len(rows),0)
        np.testing.assert_equal(P['context'][rows],P2['context'][rows2])
        model=FieldCalibrator();act=callback(model,cfg);ids=P['ids'][rows]
        a=act(P,rows,state['inventory'][ids],state['pending'][ids].sum(1))
        b=act(P2,rows2,state['inventory'][ids],state['pending'][ids].sum(1))
        np.testing.assert_equal(a,b)
        # Full-panel fingerprint protects validation episode data as well as old labels.
        self.assertNotEqual(panel_digest(p),panel_digest(p2))
        p3={k:v.copy() for k,v in p.items()};p3['demand'][0,350]+=1
        self.assertNotEqual(panel_digest(p),panel_digest(p3))

    def test_invalid_closed_form_or_mismatched_features_rejected(self):
        P=plan(self.p,370,30,self.cfg,lead_scale=2)
        with self.assertRaisesRegex(ValueError,'Closed-form OPT'):
            simulate(self.p,P,self.cfg,370,30,'OPT',lead_scale=2,shortage_mode='hybrid')
        with self.assertRaisesRegex(ValueError,'features'):
            simulate(self.p,P,self.cfg,370,30,'Normal',lead_scale=1)

    def test_autograd_matches_independent_finite_difference(self):
        p={k:v.copy() for k,v in self.p.items()};rng=np.random.default_rng(712)
        p['demand']=rng.gamma(9,2,p['demand'].shape)
        policy=FieldCalibrator().double().eval()
        with torch.no_grad():policy.net[-1].weight.fill_(.003);policy.net[-1].bias.fill_(.07)
        for lead in [1,2]:
            for mode in ['backorder','hybrid']:
                state=warm_state(p,self.cfg,300,70,300,lead,mode)
                ep=tensor_episode(episode(p,self.cfg,self.encoder,370,25,300,lead,mode,state),'cpu',torch.float64)
                policy.zero_grad(set_to_none=True);rollout_torch(policy,ep)['total_cost'].mean().backward()
                automatic=float(policy.net[-1].bias.grad[0]);original=float(policy.net[-1].bias[0]);eps=1e-6
                with torch.no_grad():
                    policy.net[-1].bias.fill_(original+eps);plus=float(rollout_torch(policy,ep)['total_cost'].mean())
                    policy.net[-1].bias.fill_(original-eps);minus=float(rollout_torch(policy,ep)['total_cost'].mean())
                    policy.net[-1].bias.fill_(original)
                self.assertGreater(abs(automatic),1e-4)
                np.testing.assert_allclose(automatic,(plus-minus)/(2*eps),rtol=1e-5,atol=1e-3)

if __name__=='__main__':unittest.main()
