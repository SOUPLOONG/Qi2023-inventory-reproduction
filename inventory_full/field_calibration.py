"""V4 engineering extension: decision-cost calibration in matched field dynamics.

Not the original paper's supervised model. The offline encoder stays frozen.
The conditional hindsight label is NOT reused for crossing/lost-sales settings.
Only observed inventory and total open quantity enter the policy, never actual
future arrival dates. Differentiable arrivals belong to the training environment.
"""
from pathlib import Path
import copy,time,json,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import norm,t as student_t
from .engine import plan,simulate
from .training import predict
from .core import csv,dump,digest

HOLDING=12.;SHORTAGE=88.

def panel_digest(p):
    """Includes calibration validation observations missing from offline samples."""
    digest_value=hashlib.sha256()
    for key in sorted(p):
        a=np.ascontiguousarray(p[key])
        digest_value.update(key.encode());digest_value.update(str(a.dtype).encode())
        digest_value.update(str(a.shape).encode());digest_value.update(a.tobytes())
    return digest_value.hexdigest()

class FieldCalibrator(nn.Module):
    def __init__(self,context_dim=149,max_adjustment=8.):
        super().__init__();self.context_dim=context_dim;self.max_adjustment=max_adjustment
        self.net=nn.Sequential(nn.LayerNorm(context_dim+3),nn.Linear(context_dim+3,64),nn.SiLU(),
            nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,context,inventory,open_quantity,scale,base_target):
        state=torch.stack([inventory/scale,open_quantity/scale,(inventory+open_quantity)/scale],dim=-1)
        correction=self.max_adjustment*torch.tanh(self.net(torch.cat([context,state],dim=-1)).squeeze(-1))
        return torch.relu(base_target+scale*correction-inventory-open_quantity)

def target_values(P,p):
    ids=P['ids'];R=p['review'][ids];mu=P['mu'];sd=P['sd'];lm=P['lm'];ls=P['ls']
    return mu*(R+lm)+norm.ppf(SHORTAGE/(HOLDING+SHORTAGE))*np.sqrt((R+lm)*sd**2+mu**2*ls**2)

def attach_context(p,P,encoder,cfg):
    encoded=predict(encoder,P['x'],cfg)['rep']
    target=target_values(P,p);scale=p['scale'][P['ids']]
    context=np.c_[encoded,P['x']['stats'],P['x']['vl'],P['x']['static'],P['x']['review'],
        target/scale,np.full(len(scale),SHORTAGE/(HOLDING+SHORTAGE))].astype('float32')
    if context.shape[1]!=149:raise AssertionError('Unexpected calibration context dimension')
    P.update(context=context,target=target,scale=scale)
    return P

def warm_state(p,cfg,start,length,anchor,lead_scale,shortage_mode):
    state={};P=plan(p,start,length,cfg,anchor=anchor,lead_scale=lead_scale)
    simulate(p,P,cfg,start,length,'Normal',h=HOLDING,b=SHORTAGE,lead_scale=lead_scale,
        shortage_mode=shortage_mode,final_state=state)
    return state

def episode(p,cfg,encoder,start,length,anchor,lead_scale,shortage_mode,initial_state):
    P=attach_context(p,plan(p,start,length,cfg,anchor=anchor,lead_scale=lead_scale),encoder,cfg)
    N=len(p['scale']);context=np.zeros((length,N,149),dtype='float32')
    target=np.zeros((length,N),dtype='float64');schedule=np.zeros((length,N),dtype=bool)
    d=P['times']-start;context[d,P['ids']]=P['context'];target[d,P['ids']]=P['target'];schedule[d,P['ids']]=True
    return dict(context=context,target=target,schedule=schedule,scale=p['scale'],
        demand=p['demand'][:,start:start+length].T,actual_lead=p['lead'][:,start:start+length].T*lead_scale,
        initial_inventory=initial_state['inventory'].copy(),initial_pending=initial_state['pending'][:,:9].copy(),
        start=start,length=length,anchor=anchor,lead_scale=lead_scale,shortage_mode=shortage_mode)

def tensor_episode(ep,device,dtype=torch.float32):
    result={}
    for k,v in ep.items():
        if isinstance(v,np.ndarray):
            dt=torch.bool if k=='schedule' else torch.long if k=='actual_lead' else dtype
            result[k]=torch.as_tensor(v,device=device,dtype=dt)
        else:result[k]=v
    return result

def rollout_torch(policy,ep,indices=None,record=False):
    """Finite-horizon cost with autograd through policy-specific inventory/queues.

    Ring queue columns are simulator-only realized arrival days. Policy sees only
    sum(queue[:,1:]); receiving today's arrivals is an observed event.
    """
    indices=torch.arange(len(ep['scale']),device=ep['scale'].device) if indices is None else indices
    I=ep['initial_inventory'][indices];queue=ep['initial_pending'][indices];scale=ep['scale'][indices]
    H=torch.zeros_like(I);B=H.clone();lostsum=H.clone();pos=H.clone();short=H.clone();states=[];actions=[]
    for dt in range(ep['length']):
        I=I+queue[:,0];future=queue[:,1:];open_quantity=future.sum(-1)
        a=policy(ep['context'][dt,indices],I,open_quantity,scale,ep['target'][dt,indices])
        a=a*ep['schedule'][dt,indices]
        # Actual lead enters the environment after action selection, not inputs.
        placement=torch.nn.functional.one_hot(ep['actual_lead'][dt,indices]-1,8).to(I.dtype)
        future=future+placement*a[:,None]
        raw=I-ep['demand'][dt,indices]
        lost=torch.relu(-raw-future.sum(-1)) if ep['shortage_mode']=='hybrid' else torch.zeros_like(raw)
        I=raw+lost;hh=HOLDING*torch.relu(I);bb=SHORTAGE*(torch.relu(-I)+lost)
        H=H+hh;B=B+bb;lostsum=lostsum+lost;pos=pos+torch.relu(I)
        short=short+((I<0)|(lost>0)).to(I.dtype)
        queue=torch.cat([future,torch.zeros_like(I[:,None])],dim=-1)
        if record:states.append(I);actions.append(a)
    result=dict(holding_cost=H,stockout_cost=B,total_cost=H+B,lost_units=lostsum,
        positive_inventory_sum=pos,stockout_days=short,final_inventory=I,final_queue=queue)
    if record:result.update(inventory=torch.stack(states),actions=torch.stack(actions))
    return result

def callback(model,cfg):
    def act(P,rows,inventory,open_quantity):
        device=cfg['device'];dtype=next(model.parameters()).dtype
        values=[P['context'][rows],inventory,open_quantity,P['scale'][rows],P['target'][rows]]
        with torch.no_grad():return model(*[torch.as_tensor(v,device=device,dtype=dtype) for v in values]).cpu().numpy()
    return act

def train_calibrator(p,cfg,out,encoder,lead_scale,shortage_mode,tag):
    epochs=int(cfg.get('field_epochs',4 if cfg['n_series']<=80 else 20))
    field_batch=int(cfg.get('field_batch_size',256));lr=float(cfg.get('field_lr',.002))
    spec=dict(version=4,lead_scale=lead_scale,shortage_mode=shortage_mode,h=HOLDING,b=SHORTAGE,
        train=[220,279],validation=[330,359],validation_environment='lead2_hybrid_for_both_calibrators',test=[370,399],epochs=epochs,batch_size=field_batch,lr=lr,
        seed=cfg['seed']+401,context_dim=149,max_adjustment=8.,
        data_signature=cfg['data_signature'],full_panel_signature=panel_digest(p),encoder_sha256=digest(out/'models/main.pt'),
        source_signature='|'.join(digest(Path(__file__).with_name(f)) for f in ['field_calibration.py','engine.py','core.py']))
    path=out/f'models/{tag}.pt'
    torch.manual_seed(spec['seed']);net=FieldCalibrator().to(cfg['device'])
    if path.exists() and not cfg.get('force',False):
        ck=torch.load(path,map_location=cfg['device'],weights_only=True)
        if ck['spec']!=spec:raise ValueError('Calibration cache differs; change output or use --force: '+tag)
        net.load_state_dict(ck['weights']);net.eval();return net,ck['metadata']
    initial=warm_state(p,cfg,200,20,200,lead_scale,shortage_mode)
    # Hold the target validation environment fixed, isolating training-domain change.
    val_initial=warm_state(p,cfg,300,30,300,2,'hybrid')
    train_ep=tensor_episode(episode(p,cfg,encoder,220,60,200,lead_scale,shortage_mode,initial),cfg['device'])
    val_ep=tensor_episode(episode(p,cfg,encoder,330,30,300,2,'hybrid',val_initial),cfg['device'])
    optimizer=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    denominator=60*100*float(p['scale'].mean()) # fixed calibration-only units; no per-series reweighting
    with torch.no_grad():
        best=float(rollout_torch(net,val_ep)['total_cost'].mean())
        train0=float(rollout_torch(net,train_ep)['total_cost'].mean())
    best_weights=copy.deepcopy(net.state_dict());best_epoch=0
    log=[dict(epoch=0,train_cost=train0,validation_cost=best,baseline_candidate=True)]
    start=time.perf_counter();rng=torch.Generator(device='cpu').manual_seed(spec['seed'])
    for e in range(1,epochs+1):
        net.train();total=0.;count=0
        for ix in torch.randperm(len(p['scale']),generator=rng).split(field_batch):
            ix=ix.to(cfg['device']);optimizer.zero_grad(set_to_none=True)
            costs=rollout_torch(net,train_ep,ix)['total_cost'];loss=costs.mean()/denominator
            if not torch.isfinite(loss):raise FloatingPointError(tag+' nonfinite rollout loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            total+=float(costs.detach().sum());count+=len(ix)
        net.eval()
        with torch.no_grad():val=float(rollout_torch(net,val_ep)['total_cost'].mean())
        log.append(dict(epoch=e,train_cost=total/count,validation_cost=val,baseline_candidate=False))
        if val<best:best=val;best_epoch=e;best_weights=copy.deepcopy(net.state_dict())
        if e==1 or e%5==0 or e==epochs:print(f'  {tag} {e}/{epochs}: validation cost={val:.2f}',flush=True)
    net.load_state_dict(best_weights);net.eval()
    metadata=dict(spec=spec,best_epoch=best_epoch,best_validation_cost=best,train_seconds=time.perf_counter()-start,
        training_series=len(p['scale']),baseline_epoch0_cost=log[0]['validation_cost'],
        source='SYNTHETIC engineering extension, not original Qi supervised architecture')
    torch.save(dict(weights=best_weights,spec=spec,metadata=metadata),path)
    dump(out/f'logs/{tag}_metadata.json',metadata);csv(pd.DataFrame(log),out/f'tables/train_{tag}.csv')
    return net,metadata

def evaluate_policies(p,cfg,out,encoder,transfer,offline,matched):
    rows=[];all_series=[];all_P={};all_states={}
    # Factorial environment comparison, no settings chosen using test costs.
    for lead_scale in [1,2]:
        for mode in ['backorder','hybrid']:
            name=f'lead{lead_scale}_{mode}'
            state=warm_state(p,cfg,300,70,300,lead_scale,mode)
            P=attach_context(p,plan(p,370,30,cfg,anchor=300,lead_scale=lead_scale),encoder,cfg)
            pred=predict(transfer,P['x'],cfg)
            frames=[]
            for method,policy in [('Normal',None),('V3_Transfer',None),('V4_OfflineCalibrator',offline),('V4_MatchedCalibrator',matched)]:
                mechanism='Normal' if method=='Normal' else 'E2E_RNN' if method=='V3_Transfer' else 'E2E_calibrated'
                df,trace,orders,_=simulate(p,P,cfg,370,30,mechanism,model=transfer,pred=pred,h=HOLDING,b=SHORTAGE,
                    lead_scale=lead_scale,shortage_mode=mode,initial_state=state,
                    policy=callback(policy,cfg) if policy is not None else None,keep=lead_scale==2 and mode=='hybrid')
                df['policy']=method;df['environment']=name;frames.append(df)
                if lead_scale==2 and mode=='hybrid':
                    csv(trace,out/f'trajectories/v4_field_{method}_daily.csv.gz');csv(orders,out/f'decisions/v4_field_{method}_orders.csv.gz')
            per=pd.concat(frames,ignore_index=True);all_series.append(per)
            baseline=per[per.policy=='Normal'].set_index('series_id').total_cost
            for method,g in per.groupby('policy',sort=False):
                g=g.set_index('series_id');diff=g.total_cost-baseline
                se=float(diff.std(ddof=1)/np.sqrt(len(diff)));critical=float(student_t.ppf(.975,len(diff)-1))
                rows.append(dict(environment=name,policy=method,holding_cost=g.holding_cost.mean(),stockout_cost=g.stockout_cost.mean(),
                    total_cost=g.total_cost.mean(),stockout_ratio=g.stockout_ratio.mean(),
                    relative_to_Normal_pct=100*(g.total_cost.mean()/baseline.mean()-1),
                    paired_difference=diff.mean(),paired_ci_low=diff.mean()-critical*se,paired_ci_high=diff.mean()+critical*se,n_series=len(g)))
            all_P[name]=P;all_states[name]=state
    summary=pd.DataFrame(rows);csv(summary,out/'tables/field_v4_factorial.csv')
    csv(pd.concat(all_series,ignore_index=True),out/'tables/field_v4_by_series.csv.gz')
    return summary,all_P['lead2_hybrid'],all_states['lead2_hybrid']

def run(p,cfg,out,encoder,transfer):
    offline,om=train_calibrator(p,cfg,out,encoder,1,'backorder','field_offline_control')
    matched,mm=train_calibrator(p,cfg,out,encoder,2,'hybrid','field_matched')
    summary,P,state=evaluate_policies(p,cfg,out,encoder,transfer,offline,matched)
    from .field import run as statistics
    from .figures import matching
    matching(statistics(p,cfg,out,transfer,treatment_policy=callback(matched,cfg),post_plan=P,
        treatment_label='V4_MatchedCalibrator'),out)
    # Compare differentiable training dynamics to the independent NumPy evaluator.
    ep=tensor_episode(episode(p,cfg,encoder,370,30,300,2,'hybrid',state),cfg['device'])
    with torch.no_grad():torch_cost=rollout_torch(matched,ep)['total_cost'].cpu().numpy()
    per=pd.read_csv(out/'tables/field_v4_by_series.csv.gz')
    np_cost=per[(per.environment=='lead2_hybrid')&(per.policy=='V4_MatchedCalibrator')].sort_values('series_id').total_cost.to_numpy()
    np.testing.assert_allclose(torch_cost,np_cost,rtol=2e-5,atol=.1)
    dump(out/'logs/field_v4_protocol.json',dict(version=4,source='SYNTHETIC field calibration engineering extension',
        matching_environment={'train':'lead2 / hybrid','validation':'lead2 / hybrid','test':'lead2 / hybrid'},
        cost_training_days=[220,279],upstream_encoder_information_ends=329,calibration_validation_days=[330,359],test_days=[370,399],
        test_used_for_selection=False,baseline_candidate_epoch=0,objective='actual rolling holding + stockout cost, differentiable queues',
        online_state='observed net stock and total open quantity; realized future arrival dates withheld',
        frozen_encoder='original offline MQRNN main; weights selected no later than day329',
        normal_residual_policy=True,max_adjustment_in_scale_units=8,field_calibration_uses_closed_form_labels=False,
        matched_best_epoch=mm['best_epoch'],offline_control_best_epoch=om['best_epoch'],
        numpy_torch_max_cost_difference=float(np.max(abs(torch_cost-np_cost))),
        observed_true_demand_assumption=True,all_policies_share_initial_state_and_demand=True,
        attribution_limit='v3 to v4 changes policy structure and training; offline vs matched calibrator holds architecture and field-validation environment fixed while changing training dynamics',
        original_paper_exact_replication=False))
    plot(summary,out)
    return summary

def plot(summary,out):
    import matplotlib.pyplot as plt
    from .figures import save
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    field=summary[summary.environment=='lead2_hybrid']
    labels=['Normal','V3 transfer','V4 offline-trained','V4 matched-trained']
    axes[0].bar(labels,field.holding_cost,label='Holding')
    axes[0].bar(labels,field.stockout_cost,bottom=field.holding_cost,label='Stockout')
    axes[0].tick_params(axis='x',rotation=20);axes[0].set_ylabel('30-day mean cost');axes[0].legend()
    for method,g in summary.groupby('policy',sort=False):
        axes[1].plot(g.environment,g.relative_to_Normal_pct,'o-',label=method)
    axes[1].axhline(0,color='black',lw=.7);axes[1].tick_params(axis='x',rotation=20)
    axes[1].set_ylabel('Difference vs same-environment Normal (%)');axes[1].legend(fontsize=8)
    fig.suptitle('V4 SYNTHETIC calibration: actual paired test results; engineering extension')
    fig.tight_layout();save(fig,out,'field_v4_comparison')
