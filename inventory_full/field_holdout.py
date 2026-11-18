"""Fixed independent synthetic panels, evaluation only; never tune checkpoints here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t
from .data import generate
from .engine import plan,simulate
from .models import InventoryNet
from .training import predict
from .field_calibration import FieldCalibrator,attach_context,warm_state,callback
from .core import csv,dump,digest

SEEDS=[20261020,20261021,20261022]

def run(cfg,out):
    root=Path(__file__).resolve().parents[1]
    # This declaration is persisted before any independent outcomes are evaluated.
    declaration=dict(seeds=SEEDS,n_series_each=80 if cfg['n_series']<=80 else 1000,days=420,
        evaluated_days=[370,399],environment='lead2_hybrid',purpose='evaluation only',
        selection_or_retraining=False,source='SYNTHETIC, not JD data')
    dump(out/'logs/independent_holdout_plan.json',declaration)
    def original(tag):
        ck=torch.load(out/f'models/{tag}.pt',map_location=cfg['device'],weights_only=True);s=ck['spec']
        m=InventoryNet(H=s['H'],hidden=s['hidden'],layers=s['layers']).to(cfg['device']);m.load_state_dict(ck['weights']);m.eval();return m
    encoder=original('main');transfer=original('field_ratio');policies={}
    for tag in ['field_offline_control','field_matched']:
        ck=torch.load(out/f'models/{tag}.pt',map_location=cfg['device'],weights_only=True)
        m=FieldCalibrator().to(cfg['device']);m.load_state_dict(ck['weights']);m.eval();policies[tag]=m
    fingerprint={p.name:digest(p) for p in (out/'models').glob('*.pt')}
    summaries=[];cohorts=[]
    for seed in SEEDS:
        c=dict(cfg,seed=seed,n_series=declaration['n_series_each'],days=420)
        data=root/'data'/f'independent_{c["n_series"]}_{seed}'
        if (data/'panel.npz').exists():
            prov=json.loads((data/'provenance.json').read_text('utf-8'))
            if prov['panel_sha256']!=digest(data/'panel.npz'):raise ValueError('Independent panel checksum differs')
            with np.load(data/'panel.npz') as z:p={k:z[k] for k in z.files}
        else:p=generate(data,c)
        state=warm_state(p,c,300,70,300,2,'hybrid')
        P=attach_context(p,plan(p,370,30,c,anchor=300,lead_scale=2),encoder,c)
        pred=predict(transfer,P['x'],c);frames=[]
        for method,policy in [('Normal',None),('V3_Transfer',None),('V4_OfflineCalibrator',policies['field_offline_control']),('V4_MatchedCalibrator',policies['field_matched'])]:
            mechanism='Normal' if method=='Normal' else 'E2E_RNN' if method=='V3_Transfer' else 'E2E_calibrated'
            df,*_=simulate(p,P,c,370,30,mechanism,model=transfer,pred=pred,h=12,b=88,lead_scale=2,
                shortage_mode='hybrid',initial_state=state,policy=callback(policy,c) if policy is not None else None)
            df['policy']=method;df['panel_seed']=seed;frames.append(df)
        per=pd.concat(frames,ignore_index=True);cohorts.append(per)
        base=per[per.policy=='Normal'].set_index('series_id').total_cost
        for method,g in per.groupby('policy',sort=False):
            g=g.set_index('series_id');delta=g.total_cost-base;se=delta.std(ddof=1)/np.sqrt(len(delta));crit=student_t.ppf(.975,len(delta)-1)
            summaries.append(dict(panel_seed=seed,policy=method,n_series=len(g),total_cost=g.total_cost.mean(),
                holding_cost=g.holding_cost.mean(),stockout_cost=g.stockout_cost.mean(),stockout_ratio=g.stockout_ratio.mean(),
                relative_to_Normal_pct=100*(g.total_cost.mean()/base.mean()-1),paired_difference=delta.mean(),
                paired_ci_low=delta.mean()-crit*se,paired_ci_high=delta.mean()+crit*se))
        print(f'[independent holdout] seed={seed} evaluated without fitting',flush=True)
    assert fingerprint=={p.name:digest(p) for p in (out/'models').glob('*.pt')},'Evaluation changed model weights'
    csv(pd.DataFrame(summaries),out/'tables/independent_holdout_summary.csv')
    csv(pd.concat(cohorts,ignore_index=True),out/'tables/independent_holdout_by_series.csv.gz')
    dump(out/'logs/independent_holdout_validation.json',dict(passed=True,weights_unchanged=True,model_sha256=fingerprint,
        n_panels=len(SEEDS),n_series_per_panel=declaration['n_series_each'],not_used_for_selection=True))
