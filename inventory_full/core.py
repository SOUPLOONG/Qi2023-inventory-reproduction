from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csv(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig',compression='infer')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def label(demand,stock,lead,next_arrival,b=9.,h=1.):
    """Conditional hindsight action at ORDER date; no other pending arrival before lead.
    Demand[0] is order-date demand. Arrivals occur before that day's demand.
    """
    if not 0<lead<next_arrival<=len(demand):raise ValueError('Invalid arrival interval')
    k=lead+min(next_arrival-lead-1,int(np.floor(b*(next_arrival-lead)/(h+b)+1e-12)))
    return max(float(np.sum(demand[:k+1]))-float(stock),0.)

def interval_target(path,lead,n,b=9.,h=1.):
    k=int(np.floor(b*n/(h+b)+1e-12))
    return float(np.sum(path[:lead+k+1]))

def cost(inv,h=1.,b=9.):
    inv=np.asarray(inv,float)
    H=h*np.maximum(inv,0).sum(axis=-1);B=b*np.maximum(-inv,0).sum(axis=-1)
    return H,B,H+B

def advance(stock,arrivals,demand,pending,h=1.,b=9.,mode='backorder'):
    """Full backlog or explicitly documented hybrid extension.
    Hybrid: backlog limited by outstanding replenishment; excess demand is lost.
    Lost units penalized once; backlog penalized each period.
    """
    raw=stock+arrivals-demand
    lost=np.maximum(-raw-pending,0) if mode=='hybrid' else np.zeros_like(raw)
    inv=raw+lost
    H=h*np.maximum(inv,0);B=b*(np.maximum(-inv,0)+lost)
    return inv,H,B,lost
