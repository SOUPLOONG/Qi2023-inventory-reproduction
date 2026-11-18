"""Synthetic panel + causal feature construction + matured-label temporal splits."""
import numpy as np
import pandas as pd
from .core import dump,csv,label,digest

Q=np.array([.1,.5,.6,.7,.8,.9,.95],dtype=np.float32)

def generate(root,cfg):
    root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(cfg['seed']);N=cfg['n_series'];D=cfg['days']
    category=rng.integers(0,6,N);warehouse=rng.integers(0,4,N)
    base=rng.lognormal(2.4,.65,N);phase=rng.uniform(0,2*np.pi,N)
    amp=rng.uniform(.12,.45,N);trend=rng.uniform(-.0005,.001,N)
    dates=np.arange(D);week=2*np.pi*dates/7
    promo=rng.binomial(1,.12,(N,D)).astype(np.float32)
    rate=base[:,None]*(1+amp[:,None]*np.sin(week+phase[:,None]))*(1+.8*promo)
    rate*=np.exp(trend[:,None]*(dates-180))
    rate*=1+.15*np.sin(2*np.pi*dates/90)[None,:]
    demand=rng.poisson(rng.gamma(10,rate/10)).astype(np.float32)
    reliability=rng.uniform(.15,.85,N)
    lead=(1+rng.binomial(3,reliability[:,None],(N,D))).astype(np.int16)
    review=rng.integers(5,11,N).astype(np.int16)
    scale=np.maximum(demand[:,:120].mean(axis=1),.5).astype(np.float32)
    static=np.c_[np.eye(6)[category],np.eye(4)[warehouse],np.log1p(scale)/5,rng.uniform(.1,1,N)].astype(np.float32)
    panel=dict(demand=demand,promo=promo,lead=lead,review=review,scale=scale,static=static,
               category=category,warehouse=warehouse,seed=np.array(cfg['seed']))
    np.savez_compressed(root/'panel.npz',**panel)
    sid=np.repeat(np.arange(N),D);day=np.tile(dates,N)
    csv(pd.DataFrame({'series_id':sid,'day':day,'demand':demand.ravel(),'planned_promotion':promo.ravel()}),root/'daily_panel.csv.gz')
    order_rows=[]
    for i in range(N):
        for t in range(0,D-8,int(review[i])):
            order_rows.append((i,t,int(lead[i,t]),t+int(lead[i,t])))
    csv(pd.DataFrame(order_rows,columns=['series_id','potential_order_day','actual_lead','actual_arrival']),root/'potential_orders.csv.gz')
    dump(root/'provenance.json',{'source':'SYNTHETIC; no JD raw data used','n_series':N,'days':D,'daily_rows':N*D,
        'seed':cfg['seed'],'panel_sha256':digest(root/'panel.npz'),
        'scale_calibration_days':[0,119],'first_training_day':180,
        'process':'Gamma-Poisson seasonal demand; planned promotions; exogenous integer vendor leads',
        'offline_no_crossover':'review>=5, lead<=4','raw_future_leads':'labels/evaluation only, not online inputs'})
    return panel

def known_cov(p,ids,times,H):
    tt=np.asarray(times)[:,None]+np.arange(H)[None,:]
    ii=np.asarray(ids)[:,None]
    return np.stack([p['promo'][ii,tt],np.sin(2*np.pi*tt/7),np.cos(2*np.pi*tt/7)],axis=-1).astype(np.float32)

def features(p,ids,times,stocks,H=20,field=False):
    ids=np.asarray(ids,dtype=int);times=np.asarray(times,dtype=int)
    ix=times[:,None]-28+np.arange(28)[None,:];s=p['scale'][ids]
    d=p['demand'][ids[:,None],ix]
    seq=np.stack([d/s[:,None],p['promo'][ids[:,None],ix],np.sin(2*np.pi*ix/7)],axis=-1)
    # Vendor observations from t-10..t-5 are all completed by t, even with lead<=4.
    # Field scaled leads need an earlier safe observation lag (max=8).
    lag=14 if field else 10
    li=times[:,None]-lag+np.arange(6)[None,:]
    lv=p['lead'][ids[:,None],li].astype(float)*(2 if field else 1)
    vl=np.c_[lv/4,lv.mean(1)/4,lv.std(1)/4,lv.min(1)/4,lv.max(1)/4,
              np.median(lv,axis=1)/4,(lv[:,3:].mean(1)-lv[:,:3].mean(1))/4]
    norm=d/s[:,None]
    basic=[norm.mean(1),norm.std(1),np.median(norm,axis=1),np.quantile(norm,.1,axis=1),
           np.quantile(norm,.25,axis=1),np.quantile(norm,.75,axis=1),np.quantile(norm,.9,axis=1),
           (d==0).mean(1),norm.std(1)/(norm.mean(1)+.1),(norm[:,-7:].mean(1)-norm[:,:7].mean(1))]
    for w in [7,14,28]:
        a=norm[:,-w:];basic.extend([a.mean(1),a.std(1),a.min(1),a.max(1)])
    stats=np.stack(basic,axis=1)
    forks=[20,24,28]
    future=np.stack([known_cov(p,ids,times-28+k,H) for k in forks],axis=1)
    out=dict(seq=seq.astype('float32'),static=p['static'][ids].astype('float32'),vl=vl.astype('float32'),
             stats=stats.astype('float32'),known=future,review=p['review'][ids].astype('float32')/10,
             stock=np.asarray(stocks,dtype='float32')/s)
    assert out['vl'].shape[1]==12 and out['stats'].shape[1]==22
    assert 28*3+12+12+22+2==132
    return out

def labels_for(ds,b,h=1):
    n=ds['R']+ds['Lnext']-ds['L']
    cutoff=ds['L']+np.floor(b*n/(b+h)+1e-12).astype(int)
    cumulative=np.cumsum(ds['actual_future'],axis=1)
    return np.maximum(cumulative[np.arange(len(cutoff)),cutoff]-ds['I'],0).astype('float32')/ds['scale']

def build_samples(p,cfg,root):
    """States from a causal NORMAL behavior policy, then conditional hindsight labels.
    They are not claimed to be the states of a clairvoyant training trajectory.
    """
    rng=np.random.default_rng(cfg['seed']+1);H=cfg['horizon'];N=len(p['scale'])
    records=[];meta=[]
    split_bounds={'train':(180,280),'val':(280,330)}
    for i in range(N):
        R=int(p['review'][i]);I=float(p['scale'][i]*R);arrivals={}
        for t in range(180,330):
            I+=arrivals.pop(t,0.)
            if (t-180)%R==0:
                L=int(p['lead'][i,t]);Ln=int(p['lead'][i,t+R])
                split='train' if t<280 else 'val'
                if t+H<=split_bounds[split][1]:
                    candidate=[I]
                    if split=='train':candidate.append(float(rng.uniform(-3,R+4)*p['scale'][i]))
                    for stock in candidate:
                        records.append((i,t,stock));meta.append((i,t,stock,L,Ln,R,split))
                past=p['demand'][i,t-180:t];mu=float(past.mean());sd=float(past.std())
                lhist=p['lead'][i,t-10:t-4]
                target=mu*(R+lhist.mean())+1.2815515655*np.sqrt((R+lhist.mean())*sd**2+mu**2*lhist.var())
                a=max(target-I,0.);arrivals[t+L]=arrivals.get(t+L,0.)+a
            I-=float(p['demand'][i,t])
    ids=np.array([r[0] for r in records]);times=np.array([r[1] for r in records]);stocks=np.array([r[2] for r in records])
    ds=features(p,ids,times,stocks,H)
    ds.update(sid=ids.astype('int32'),day=times.astype('int16'),I=stocks.astype('float32'),scale=p['scale'][ids],
        L=np.array([m[3] for m in meta],dtype='int16'),Lnext=np.array([m[4] for m in meta],dtype='int16'),
        R=p['review'][ids],split=np.array([m[6] for m in meta]))
    ds['actual_future']=p['demand'][ids[:,None],times[:,None]+np.arange(H)[None,:]]
    ds['fork_y']=np.stack([p['demand'][ids[:,None],times[:,None]-28+k+np.arange(H)[None,:]] for k in [20,24,28]],axis=1)/ds['scale'][:,None,None]
    ds['label']=labels_for(ds,9)
    np.savez_compressed(root/'training_samples.npz',**ds)
    csv(pd.DataFrame({'sample_id':np.arange(len(ids)),'series_id':ids,'day':times,'net_stock':stocks,
        'actual_lead_LABEL_ONLY':ds['L'],'next_lead_LABEL_ONLY':ds['Lnext'],'review':ds['R'],
        'split':ds['split'],'order_label':ds['label']*ds['scale']}),root/'training_index.csv.gz')
    dump(root/'sample_audit.json',{'n_samples':len(ids),'split_counts':pd.Series(ds['split']).value_counts().to_dict(),
        'snapshot_dimensions':132,'known_future_plan_dimensions':3*H,
        'train_label_latest_day':int((ds['day'][ds['split']=='train']+H-1).max()),
        'validation_label_latest_day':int((ds['day'][ds['split']=='val']+H-1).max()),
        'purge':'Whole forecast horizon must finish before next temporal split',
        'state_policy':'causal Normal behavior plus future-independent random stock augmentation',
        'forks':'encoder positions 20,24,28; shared decoder; calibration scale uses days0..119 only'})
    return ds
