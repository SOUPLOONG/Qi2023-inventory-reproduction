"""Explicit method reconstruction on SYNTHETIC independent decision episodes.

NOT the authors' MQRNN source, original training data or published trained model.
The architecture preserves separate demand/lead blocks, late R/I inputs and 3 heads.
"""
from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
from .common import dump_json,save_csv
from .inventory import label_from_order_time,simulate_backorders


def make_data(root,n=3600,seed=20261019):
    """Independent episodes: 28 past days, 14 future days, R in 4..8, L in 1..3.

    No other outstanding orders. Future data and realized L are labels/evaluation
    only. Splits share no episode. This is NOT the real JD training dataset.
    """
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    history=[];basic=[];lead_features=[];future=[];initial=[];review=[];lead=[];next_arrival=[];labels=[];scales=[]
    probabilities=np.array([[.7,.2,.1],[.2,.6,.2],[.1,.2,.7],[.35,.3,.35]])
    for _ in range(n):
        category=int(rng.integers(4));base=float(rng.uniform(2,20));phase=float(rng.uniform(0,2*np.pi))
        days=np.arange(-28,14)
        promotion=rng.binomial(1,.12,size=42)
        rate=base*(1+.25*np.sin(2*np.pi*days/7+phase))*(1+.5*promotion)
        d=rng.poisson(rng.gamma(8,rate/8)).astype(float)
        scale=max(.5,float(d[:28].mean()))
        lhist=rng.choice([1,2,3],size=6,p=probabilities[category]).astype(float)
        l1,l2=rng.choice([1,2,3],size=2,p=probabilities[category]);R=int(rng.integers(4,9))
        I=float(rng.uniform(-2,10)*scale)
        lab=label_from_order_time(d[28:],I,int(l1),R+int(l2))
        history.append(np.column_stack([d[:28]/scale,promotion[:28]]))
        basic.append(np.eye(4)[category]);lead_features.append(lhist/3)
        future.append(d[28:]);initial.append(I);review.append(R);lead.append(l1)
        next_arrival.append(R+l2);labels.append(lab['order']);scales.append(scale)
    # Random split of independent simulated episodes; no temporal overlap to purge.
    order=rng.permutation(n);split=np.empty(n,dtype='U5')
    split[order[:int(.7*n)]]='train';split[order[int(.7*n):int(.85*n)]]='val';split[order[int(.85*n):]]='test'
    arrays=dict(history=np.array(history,dtype='float32'),basic=np.array(basic,dtype='float32'),
        lead_features=np.array(lead_features,dtype='float32'),future=np.array(future,dtype='float32'),
        initial=np.array(initial,dtype='float32'),review=np.array(review,dtype='int64'),
        lead=np.array(lead,dtype='int64'),next_arrival=np.array(next_arrival,dtype='int64'),
        labels=np.array(labels,dtype='float32'),scale=np.array(scales,dtype='float32'),split=split)
    np.savez_compressed(root/'synthetic_episodes.npz',**arrays)
    meta=pd.DataFrame({'sample_id':np.arange(n),'split':split,'initial_stock':initial,'review_period':review,
        'realized_lead_LABEL_ONLY':lead,'next_arrival_LABEL_ONLY':next_arrival,'scale_from_past':scales,'order_label':labels})
    save_csv(meta,root/'synthetic_samples.csv')
    # Human-readable arrays in a CSV; the NPZ is the lossless, efficient input.
    import json
    visible=meta.copy()
    for col in ['history','basic','lead_features','future']:
        visible[col]=[json.dumps(v.tolist(),separators=(',',':')) for v in arrays[col]]
    save_csv(visible,root/'synthetic_episodes_readable.csv')
    dump_json(root/'provenance.json',{'kind':'SYNTHETIC, generated for method demonstration',
        'seed':seed,'n':n,'split_counts':meta.split.value_counts().to_dict(),
        'known_inputs':['history','basic','lead_features','initial','review','scale'],
        'not_online_inputs':['future','lead','next_arrival','labels'],
        'assumptions':'independent one-decision episodes, full backorders, no other in-transit orders, no crossing',
        'costs':{'h':1,'b':9},'author_data_used':False})
    return arrays


def model_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ModularInventoryNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.demand_gru=nn.GRU(6,50,batch_first=True)
            self.demand_head=nn.Linear(50,14*3)
            self.lead_block=nn.Sequential(nn.Linear(10,50),nn.ReLU(),nn.Dropout(.2),nn.Linear(50,20),nn.ReLU())
            self.lead_head=nn.Linear(20,1)
            self.integration3=nn.Sequential(nn.Linear(50+20+1,100),nn.ReLU(),nn.Dropout(.2))
            self.integration4=nn.Sequential(nn.Linear(100+1,100),nn.ReLU(),nn.Dropout(.2))
            self.order_head=nn.Linear(100,1)
            for name,p in self.named_parameters():
                if 'weight' in name:nn.init.normal_(p,mean=0,std=.01)
                else:nn.init.zeros_(p)

        def forward(self,history,basic,lead_features,review,initial):
            static=basic[:,None,:].expand(-1,history.size(1),-1)
            _,hidden=self.demand_gru(torch.cat([history,static],dim=2))
            dh=hidden[-1]
            lh=self.lead_block(torch.cat([lead_features,basic],dim=1))
            # Positive monotone quantiles; a transparent reconstruction choice.
            demand=torch.cumsum(F.softplus(self.demand_head(dh).reshape(-1,14,3)),dim=2)
            lead=(1+2*torch.sigmoid(self.lead_head(lh).squeeze(-1)))/3
            z=self.integration3(torch.cat([dh,lh,review[:,None]],dim=1))
            z=self.integration4(torch.cat([z,initial[:,None]],dim=1))
            order=F.softplus(self.order_head(z).squeeze(-1))
            return order,demand,lead
    return ModularInventoryNet


def run(package_root,out,epochs=30,seed=20261019,n=3600):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    data=make_data(package_root/'data/synthetic',n,seed)
    scale=data['scale']
    # Only the named observable arrays enter forward(). Future/realized lead do not.
    names=['history','basic','lead_features','review','initial']
    features=[torch.tensor(data['history']),torch.tensor(data['basic']),torch.tensor(data['lead_features']),
              torch.tensor(data['review']/8,dtype=torch.float32),torch.tensor(data['initial']/scale)]
    target_order=torch.tensor(data['labels']/scale)
    target_demand=torch.tensor(data['future']/scale[:,None])
    target_lead=torch.tensor(data['lead']/3,dtype=torch.float32)
    ids={s:torch.tensor(np.flatnonzero(data['split']==s)) for s in ['train','val','test']}
    net=model_class()()
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    q=torch.tensor([.1,.5,.9])
    lambda_demand,lambda_lead=.1,.1

    def objective(pred,ix):
        order,demand,lead=pred
        err=target_demand[ix,:,None]-demand
        pinball=torch.maximum(q*err,(q-1)*err).mean()
        return F.smooth_l1_loss(order,target_order[ix])+lambda_demand*pinball+lambda_lead*F.mse_loss(lead,target_lead[ix])

    logs=[];best=float('inf');best_state=None;best_epoch=0
    for epoch in range(1,epochs+1):
        net.train();total=0
        permutation=ids['train'][torch.randperm(len(ids['train']))]
        for ix in permutation.split(64):
            optimizer.zero_grad();loss=objective(net(*[x[ix] for x in features]),ix)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
            total+=float(loss.detach())*len(ix)
        net.eval()
        with torch.no_grad():validation=float(objective(net(*[x[ids['val']] for x in features]),ids['val']))
        logs.append({'epoch':epoch,'train_loss':total/len(ids['train']),'validation_loss':validation})
        if validation<best:
            best=validation;best_state=copy.deepcopy(net.state_dict());best_epoch=epoch
        if epoch%5==0 or epoch==1:print(f'  Neural demo epoch {epoch}/{epochs}: validation joint loss={validation:.4f}',flush=True)
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():order_hat,demand_hat,lead_hat=net(*[x[ids['test']] for x in features])
    test_ix=ids['test'].numpy();pred=order_hat.numpy()*scale[test_ix]
    demand_pred=demand_hat.numpy()*scale[test_ix,None,None]
    predicted_lead=lead_hat.numpy()*3
    records=[];prediction_rows=[]
    for k,i in enumerate(test_ix):
        L=int(data['lead'][i]);vnext=int(data['next_arrival'][i]);R=int(data['review'][i]);I=float(data['initial'][i])
        forecast_lead=int(np.clip(round(float(predicted_lead[k])),1,3))
        bm2=label_from_order_time(demand_pred[k,:,1],I,forecast_lead,R+forecast_lead)['order']
        bm1=max(float(demand_pred[k,:R+forecast_lead,2].sum())-I,0)
        simple_mean=max(float(data['history'][i,:,0].mean()*scale[i])*(R+forecast_lead)-I,0)
        actions={'hindsight_OPT':float(data['labels'][i]),'modular_E2E_demo':float(pred[k]),
                 'PTO_median_path_demo':bm2,'PTO_high_quantile_demo':bm1,'mean_protection_demo':simple_mean}
        for method,a in actions.items():
            arrivals=np.zeros(vnext);arrivals[L]=a
            _,c=simulate_backorders(data['future'][i,:vnext],I,arrivals)
            records.append({'sample_id':int(i),'method':method,'order':a,**c})
        prediction_rows.append({'sample_id':int(i),'label_order':float(data['labels'][i]),'predicted_order':float(pred[k]),
            'realized_lead_LABEL_ONLY':L,'predicted_lead':float(predicted_lead[k]),'initial_stock':I,
            'review_period':R,'next_arrival_LABEL_ONLY':vnext,'forecast_lead_for_PTO':forecast_lead})
    result=pd.DataFrame(records)
    # 事后OPT是同一初始库存、同一真实路径和同一成本下的参照。
    wide=result.pivot(index='sample_id',columns='method',values='total_cost')
    for col in wide:
        if col!='hindsight_OPT' and (wide[col]<wide.hindsight_OPT-2e-3).any():
            raise AssertionError('A policy beat the deterministic oracle: inspect time indexing')
    summary=result.groupby('method')[['holding_cost','stockout_cost','total_cost']].mean().reset_index()
    summary['scope']='SYNTHETIC independent episodes; reconstruction, not published experiment'
    save_csv(result,out/'tables/neural_demo_by_episode.csv')
    save_csv(summary,out/'tables/neural_demo_summary.csv')
    save_csv(pd.DataFrame(prediction_rows),out/'tables/neural_demo_predictions.csv')
    log=pd.DataFrame(logs);save_csv(log,out/'tables/neural_demo_training.csv')
    checkpoint=out/'models/modular_demo.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    import hashlib
    dataset_hash=hashlib.sha256((package_root/'data/synthetic/synthetic_episodes.npz').read_bytes()).hexdigest()
    torch.save({'state_dict':best_state,'quantiles':[.1,.5,.9],'seed':seed,'best_epoch':best_epoch,
        'scope':'synthetic reconstruction','history_length':28,'forecast_horizon':14,
        'dataset_sha256':dataset_hash,'n_samples':n},checkpoint)
    metrics={'scope':'SYNTHETIC reconstruction; not authors trained weights',
        'seed':seed,'epochs':epochs,'best_epoch':best_epoch,'validation_best_joint_loss':best,
        'train_n':len(ids['train']),'validation_n':len(ids['val']),'test_n':len(ids['test']),
        'order_MAE':float(np.mean(np.abs(pred-data['labels'][test_ix]))),
        'lead_MAE':float(np.mean(np.abs(predicted_lead-data['lead'][test_ix]))),
        'elapsed_seconds':time.perf_counter()-start,'device':'cpu','threads':1,
        'known_forward_inputs':names,'optimizer':'Adam(lr=0.001), reconstruction choice',
        'loss':'SmoothL1(normalized order) + 0.1*pinball(normalized demand) + 0.1*MSE(lead/3)',
        'architecture':'GRU50 + direct multi-horizon quantile head; VLT50/20; integration100/100; late R/I',
        'differences':'Simplified quantile decoder, independent episodes, synthetic data, explicit substitute losses; not exact MQRNN or rolling multiorder evaluation.'}
    dump_json(out/'logs/neural_demo.json',metrics)
    from .plots import learning_chart
    learning_chart(log,out/'figures/neural_demo_learning.png')
    return summary
