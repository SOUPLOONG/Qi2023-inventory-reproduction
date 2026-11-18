from pathlib import Path
import copy,time,json,random
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .models import InventoryNet
from .data import labels_for,Q
from .core import dump,csv,digest

INPUTS=['seq','static','vl','stats','known','review','stock']

def fit(ds,cfg,out,tag='main',hidden=50,layers=2,lr=.001,fraction=1.,b=9.,seed=None,forecast_only=False):
    seed=cfg['seed'] if seed is None else seed;device=cfg['device']
    spec=dict(hidden=hidden,layers=layers,lr=lr,fraction=fraction,b=b,seed=seed,forecast_only=forecast_only,
              epochs=cfg['epochs'],batch=cfg['batch_size'],optimizer=cfg['optimizer'],H=cfg['horizon'],
              data_signature=cfg['data_signature'],implementation_signature='|'.join(
                  digest(Path(__file__).with_name(f)) for f in ['models.py','training.py','data.py','core.py']))
    path=out/'models'/f'{tag}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    model=InventoryNet(H=cfg['horizon'],hidden=hidden,layers=layers).to(device)
    if path.exists() and not cfg.get('force',False):
        ck=torch.load(path,map_location=device,weights_only=True)
        if ck['spec']!=spec:raise ValueError('Cached specification differs: '+tag)
        model.load_state_dict(ck['weights']);model.eval();return model,ck['metadata']
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    # Initialization occurs after setting seeds, ensuring fresh and cached identity.
    model=InventoryNet(H=cfg['horizon'],hidden=hidden,layers=layers).to(device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    tensor={k:torch.as_tensor(ds[k],device=device,dtype=torch.float32) for k in INPUTS}
    y=torch.as_tensor(labels_for(ds,b),device=device)
    fy=torch.as_tensor(ds['fork_y'],device=device)
    targetL=torch.as_tensor(ds['L']/4,device=device,dtype=torch.float32)
    ids=np.flatnonzero(ds['split']=='train')
    if fraction<1:
        rng=np.random.default_rng(cfg['seed']);sids=np.unique(ds['sid'][ids]);rng.shuffle(sids)
        ids=ids[np.isin(ds['sid'][ids],sids[:max(1,int(len(sids)*fraction))])]
    tr=torch.as_tensor(ids,device=device);va=torch.as_tensor(np.flatnonzero(ds['split']=='val'),device=device)
    optimizer=(torch.optim.Adam(model.parameters(),lr=lr) if cfg['optimizer']=='adam' else
               torch.optim.SGD(model.parameters(),lr=lr,momentum=.85))
    q=torch.tensor(Q,device=device);best=float('inf');best_state=None;best_epoch=0;log=[];start=time.perf_counter()
    def objective(pred,index):
        err=fy[index,...,None]-pred['daily']
        pin=torch.maximum(q*err,(q-1)*err).mean()
        er2=fy[index].cumsum(-1)[...,None]-pred['total']
        totalpin=torch.maximum(q*er2,(q-1)*er2).mean()/cfg['horizon']
        df=pin+.1*F.mse_loss(pred['mean'],fy[index])+.5*totalpin
        lt=F.mse_loss(pred['lead'],targetL[index])
        return df+lt if forecast_only else F.smooth_l1_loss(pred['order'],y[index])+.1*df+.1*lt
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;count=0
        for ix in tr[torch.randperm(len(tr),device=device)].split(cfg['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model({k:v[ix] for k,v in tensor.items()}),ix)
            if not torch.isfinite(loss):raise FloatingPointError(f'{tag} nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ix);count+=len(ix)
        model.eval();vtotal=0.
        with torch.no_grad():
            for ix in va.split(1024):vtotal+=float(objective(model({k:v[ix] for k,v in tensor.items()}),ix))*len(ix)
        val=vtotal/len(va)
        log.append({'epoch':epoch,'train_loss':total/count,'validation_loss':val})
        if val<best:best=val;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        if epoch==1 or epoch==cfg['epochs'] or epoch%5==0:
            print(f'  {tag} epoch {epoch}/{cfg["epochs"]}: val={val:.5f}',flush=True)
    model.load_state_dict(best_state);model.eval()
    metadata=dict(tag=tag,spec=spec,train_samples=len(tr),validation_samples=len(va),
        training_series=int(len(np.unique(ds['sid'][ids]))),best_epoch=best_epoch,best_validation_loss=best,
        train_seconds=time.perf_counter()-start,synthetic=True)
    torch.save({'weights':best_state,'spec':spec,'metadata':metadata},path)
    csv(pd.DataFrame(log),out/'tables'/f'train_{tag}.csv');dump(out/'logs'/f'model_{tag}.json',metadata)
    del tensor,y,fy,targetL
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return model,metadata

def predict(model,x,cfg):
    arrays={k:torch.as_tensor(x[k],device=cfg['device'],dtype=torch.float32) for k in INPUTS}
    pieces={k:[] for k in ['daily','mean','total','lead','rep']}
    model.eval()
    with torch.no_grad():
        n=len(x['stock'])
        for start in range(0,n,1024):
            pred=model({k:v[start:start+1024] for k,v in arrays.items()},forking=False)
            for k in pieces:
                value=pred[k][:,0] if k in ['daily','mean','total'] else pred[k]
                pieces[k].append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in pieces.items()}

def tree_features(x):
    return np.c_[x['static'],x['vl'],x['stats'],x['review'],x['stock']].astype('float32')

def fit_tree(ds,out,cfg,b=9.):
    import lightgbm as lgb
    tr=ds['split']=='train';va=ds['split']=='val';X=tree_features(ds);y=labels_for(ds,b)
    model=lgb.LGBMRegressor(n_estimators=500,num_leaves=31,learning_rate=.04,n_jobs=4,
        random_state=cfg['seed'],verbosity=-1,deterministic=True,force_col_wise=True)
    start=time.perf_counter()
    model.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],callbacks=[lgb.early_stopping(30,verbose=False)])
    # LightGBM native narrow-char file API fails on Windows Chinese paths.
    (out/'models/lightgbm.txt').write_text(model.booster_.model_to_string(),encoding='utf-8')
    dump(out/'logs/lightgbm.json',{'version':lgb.__version__,'actual_LightGBM':True,'features':48,
        'training_samples':int(tr.sum()),'validation_samples':int(va.sum()),
        'best_iteration':int(model.best_iteration_),'train_seconds':time.perf_counter()-start})
    return model
