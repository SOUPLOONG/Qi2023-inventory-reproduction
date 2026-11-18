"""Causal rolling policy evaluation: policy-specific inventories and arrival queues."""
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm,gamma
from .data import features,Q
from .training import predict,tree_features
from .core import label,advance,csv,dump

def plan(p,start,length,cfg,field=False,anchor=None,lead_scale=None):
    lead_scale=(2 if field else 1) if lead_scale is None else lead_scale
    if lead_scale not in [1,2]:raise ValueError('Supported lead scales: 1,2')
    anchor=start if anchor is None else anchor
    ids=[];times=[]
    for t in range(start,start+length):
        ix=np.flatnonzero((t-anchor)%p['review']==0)
        ids.extend(ix.tolist());times.extend([t]*len(ix))
    ids=np.array(ids);times=np.array(times)
    x=features(p,ids,times,np.zeros(len(ids)),cfg['horizon'],lead_scale==2)
    hist=p['demand'][ids[:,None],times[:,None]-180+np.arange(180)[None,:]]
    vendor_lag=188 if lead_scale==2 else 184
    if start<vendor_lag:raise ValueError('Insufficient completed 180-day vendor history')
    vendor=p['lead'][ids[:,None],times[:,None]-vendor_lag+np.arange(180)[None,:]]*lead_scale
    return dict(ids=ids,times=times,x=x,mu=hist.mean(1),sd=hist.std(1),
                lm=vendor.mean(1),ls=vendor.std(1),lead_scale=lead_scale)

def simulate(p,P,cfg,start,length,method,model=None,pred=None,tree=None,b=9.,h=1.,tau=5,field=False,keep=False,initial_state=None,final_state=None,lead_scale=None,shortage_mode=None,policy=None):
    lead_scale=(2 if field else 1) if lead_scale is None else lead_scale
    shortage_mode=('hybrid' if field else 'backorder') if shortage_mode is None else shortage_mode
    if lead_scale not in [1,2] or shortage_mode not in ['backorder','hybrid']:raise ValueError('Invalid environment')
    if P.get('lead_scale',lead_scale)!=lead_scale:raise ValueError('Policy features and realized lead environments differ')
    if method=='OPT' and (lead_scale!=1 or shortage_mode!='backorder'):raise ValueError('Closed-form OPT invalid for this environment')
    N=len(p['scale']);I=p['demand'][:,start-180:start].mean(1)*p['review']/2
    pending=np.zeros((N,length+20),dtype=np.float64)
    if initial_state is not None:
        I=initial_state['inventory'].copy()
        tail=initial_state['pending'];pending[:,:tail.shape[1]]=tail
    initial=I.copy()
    H=np.zeros(N);B=np.zeros(N);lostsum=np.zeros(N);pos=np.zeros(N);short=np.zeros(N)
    records=[];orders=[];inventory=np.zeros((N,length),dtype=np.float32)
    rep=None
    if pred is not None and model is not None:rep=torch.as_tensor(pred['rep'],device=cfg['device'])
    for dt,t in enumerate(range(start,start+length)):
        received=pending[:,dt].copy()
        # Arrival before order and demand. Defaults ensure no old own order arrives at a review time.
        I+=received
        rows=np.flatnonzero(P['times']==t);ids=P['ids'][rows]
        if len(ids):
            outstanding=pending[:,dt+1:].sum(1)
            position=I[ids]+outstanding[ids]
            remaining=start+length-t
            if method=='OPT':
                a=[]
                for i in ids:
                    L=int(p['lead'][i,t]);R=int(p['review'][i])
                    end=min(t+R+int(p['lead'][i,t+R]),start+length)
                    a.append(label(p['demand'][i,t:end],I[i],L,end-t,b,h) if t+L<end else 0.)
                a=np.array(a)
            elif method=='E2E_calibrated':
                # Only observed net inventory and total open quantity reach policy.
                # The simulator's realized future arrival dates are never policy inputs.
                a=policy(P,rows,I[ids].copy(),outstanding[ids].copy())
            elif method=='E2E_RNN':
                with torch.no_grad():a=model.order_from_rep(rep[rows],torch.as_tensor(position/p['scale'][ids],device=cfg['device'],dtype=torch.float32)).cpu().numpy()*p['scale'][ids]
            elif method=='E2E_GBM':
                x={k:v[rows].copy() for k,v in P['x'].items()};x['stock']=position/p['scale'][ids]
                booster=tree.booster_ if hasattr(tree,'booster_') else tree
                a=booster.predict(tree_features(x),num_threads=4)*p['scale'][ids]
            elif method in ['Normal','Gamma']:
                R=p['review'][ids];mu=P['mu'][rows];sd=P['sd'][rows];lm=P['lm'][rows];ls=P['ls'][rows]
                protection=R+lm
                if method=='Normal':target=mu*protection+norm.ppf(b/(b+h))*np.sqrt(protection*sd**2+mu**2*ls**2)
                else:
                    theta=sd**2/np.maximum(mu,.01);shape=mu/np.maximum(theta,.001)
                    target=gamma.ppf(b/(b+h),a=protection*shape,scale=np.maximum(theta,.001))
                    target=np.where(sd<.001,mu*protection,target)
                a=target-position
            elif method in ['BM1','BM2']:
                lp=np.clip(np.rint(pred['lead'][rows]*4).astype(int),1,cfg['horizon']-1)
                prot=np.clip(p['review'][ids]+lp,1,min(cfg['horizon'],remaining))
                if method=='BM1':a=pred['total'][rows,prot-1,tau]*p['scale'][ids]-position
                else:
                    a=[]
                    for j,(i,r,L,pr) in enumerate(zip(ids,rows,lp,prot)):
                        path=pred['daily'][r,:,tau]*p['scale'][i]
                        if L>=pr:a.append(0.)
                        else:a.append(label(path[:pr],position[j],int(L),int(pr),b,h))
                    a=np.array(a)
            else:raise ValueError(method)
            a=np.maximum(np.asarray(a,float),0.)
            if not np.isfinite(a).all():raise FloatingPointError('Nonfinite action: '+method)
            actual=p['lead'][ids,t].astype(int)*lead_scale
            pending[ids,dt+actual]+=a
            if keep:
                orders.extend({'method':method,'series_id':int(i),'order_day':t,'arrival_day':t+int(L),
                    'order_quantity':float(q),'net_stock_at_order':float(I[i]),'inventory_position':float(ip)}
                    for i,L,q,ip in zip(ids,actual,a,position))
        future_pending=pending[:,dt+1:].sum(1)
        I,hh,bb,lost=advance(I,np.zeros(N),p['demand'][:,t],future_pending,h,b,shortage_mode)
        H+=hh;B+=bb;lostsum+=lost;pos+=np.maximum(I,0);short+=(I<0)|(lost>0)
        inventory[:,dt]=I
        if keep:
            records.extend({'method':method,'series_id':i,'day':t,'received':float(received[i]),
                'demand':float(p['demand'][i,t]),'end_inventory':float(I[i]),'holding_cost':float(hh[i]),
                'stockout_cost':float(bb[i]),'lost_sales':float(lost[i]),'pending_after_day':float(future_pending[i])}
                for i in range(N))
    avgd=p['demand'][:,start:start+length].mean(1)
    df=pd.DataFrame({'series_id':np.arange(N),'method':method,'holding_cost':H,'stockout_cost':B,'total_cost':H+B,
        'turnover_days':pos/length/np.maximum(avgd,.001),'stockout_ratio':short/length,'lost_units':lostsum,
        'initial_stock':initial,'mean_demand':avgd})
    if final_state is not None:
        final_state.update(inventory=I.copy(),pending=pending[:,length:].copy())
    return df,pd.DataFrame(records),pd.DataFrame(orders),inventory

def choose_quantiles(p,P,cfg,forecast,start,length,b=9):
    results=[];selected={}
    for method in ['BM1','BM2']:
        for qi,q in enumerate(Q):
            df,*_=simulate(p,P,cfg,start,length,method,pred=forecast,b=b,tau=qi)
            results.append({'method':method,'quantile':float(q),'index':qi,'validation_cost':float(df.total_cost.mean())})
        best=min([r for r in results if r['method']==method],key=lambda r:r['validation_cost'])
        selected[method]=best['index']
    return selected,pd.DataFrame(results)

def evaluate_model(p,cfg,out,model,forecast_model,tree,b=9,tag='main',keep=False,selected=None):
    if selected is None:
        val=plan(p,cfg['val_start'],cfg['val_days'],cfg)
        vf=predict(forecast_model,val['x'],cfg)
        selected,selection=choose_quantiles(p,val,cfg,vf,cfg['val_start'],cfg['val_days'],b)
        csv(selection,out/'tables'/f'quantile_selection_{tag}.csv')
    P=plan(p,cfg['test_start'],cfg['test_days'],cfg)
    e=predict(model,P['x'],cfg);f=predict(forecast_model,P['x'],cfg)
    frames=[]
    methods=['OPT','E2E_RNN','BM1','BM2','Normal','Gamma','E2E_GBM'] if tag=='main' else ['OPT','E2E_RNN','Normal','Gamma']
    for method in methods:
        df,traj,orders,_=simulate(p,P,cfg,cfg['test_start'],cfg['test_days'],method,model=model,
            pred=e if method=='E2E_RNN' else f,tree=tree,b=b,tau=selected.get(method,5),keep=keep)
        frames.append(df)
        if keep:
            csv(traj,out/'trajectories'/f'{method}_daily.csv.gz');csv(orders,out/'decisions'/f'{method}_orders.csv.gz')
    per=pd.concat(frames,ignore_index=True)
    wide=per.pivot(index='series_id',columns='method',values='total_cost')
    if ((wide.drop(columns='OPT').sub(wide.OPT,axis=0)) < -1e-3).any().any():
        raise AssertionError('Clairvoyant oracle lower bound violated')
    summary=per.groupby('method',sort=False)[['holding_cost','stockout_cost','total_cost','turnover_days','stockout_ratio']].mean().reset_index()
    summary['relative_gap_pct']=(summary.total_cost/float(summary.loc[summary.method=='OPT','total_cost'].iloc[0])-1)*100
    csv(summary,out/'tables'/f'costs_{tag}.csv');csv(per,out/'tables'/f'costs_by_series_{tag}.csv')
    if tag=='main':
        rows=[]
        sample_ids=[0,1,2]  # fixed first three series; never chosen by fit quality
        for i in sample_ids:
            r=np.flatnonzero(P['ids']==i)[0];t=int(P['times'][r]);s=p['scale'][i]
            for k in range(-28,cfg['horizon']):
                rows.append({'series_id':i,'relative_day':k,'day':t+k,'demand':float(p['demand'][i,t+k]),
                    'q10':float(e['daily'][r,k,0]*s) if k>=0 else np.nan,
                    'q50':float(e['daily'][r,k,1]*s) if k>=0 else np.nan,
                    'q90':float(e['daily'][r,k,5]*s) if k>=0 else np.nan})
        csv(pd.DataFrame(rows),out/'tables/figure2_forecasts.csv')
        dump(out/'logs/rollout_protocol.json',{'source':'SYNTHETIC','n_series':len(p['scale']),'start':cfg['test_start'],
            'days':cfg['test_days'],'all_policy_specific_states':True,'a_nonnegative':True,
            'complete_day_costs':True,'no_reset_backorders':True,'BM_quantiles_selected_on_validation_only':selected,
            'OPT':'same initial stock and realized path; clairvoyant decisions only',
            'terminal':'oracle truncates last arrival interval; stationary learned/base-stock policies have no separate terminal-value training'})
    return summary,selected
