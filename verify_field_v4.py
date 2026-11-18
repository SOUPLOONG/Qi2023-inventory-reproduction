"""Audit field calibration from saved raw outputs, not visual resemblance."""
from pathlib import Path
import argparse,json,hashlib,os,sys
ROOT=Path(__file__).resolve().parent
if os.name=='nt':sys.path.insert(0,str(ROOT/'.vendor'))
import numpy as np
import pandas as pd
import torch
from inventory_full.field_calibration import panel_digest

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(profile):
    out=ROOT/f'results/{profile}';checks=[]
    def check(name,yes):
        if not yes:raise AssertionError(name)
        checks.append(name)
    cfg=json.loads((out/'config_used.json').read_text('utf-8'));N=cfg['n_series']
    with np.load(ROOT/f'data/{profile}/panel.npz') as z:p={k:z[k] for k in z.files}
    source='|'.join(sha(ROOT/'inventory_full'/f) for f in ['field_calibration.py','engine.py','core.py'])
    for tag,lead,mode in [('field_offline_control',1,'backorder'),('field_matched',2,'hybrid')]:
        meta=json.loads((out/f'logs/{tag}_metadata.json').read_text('utf-8'));spec=meta['spec']
        check(tag+' full panel identity',spec['full_panel_signature']==panel_digest(p))
        check(tag+' encoder identity',spec['encoder_sha256']==sha(out/'models/main.pt'))
        check(tag+' current source',spec['source_signature']==source)
        check(tag+' training environment',(spec['lead_scale'],spec['shortage_mode'])==(lead,mode))
        check(tag+' same target validation',spec['validation_environment']=='lead2_hybrid_for_both_calibrators')
        check(tag+' temporal boundaries',spec['train']==[220,279] and spec['validation']==[330,359] and spec['test']==[370,399])
        log=pd.read_csv(out/f'tables/train_{tag}.csv')
        np.testing.assert_allclose(meta['best_validation_cost'],log.validation_cost.min(),atol=1e-6)
        check(tag+' baseline candidate retained',log.iloc[0].epoch==0 and bool(log.iloc[0].baseline_candidate))
        ck=torch.load(out/f'models/{tag}.pt',map_location='cpu',weights_only=True)
        check(tag+' saved checkpoint metadata',ck['spec']==spec and ck['metadata']['best_epoch']==meta['best_epoch'])
        check(tag+' finite saved weights',all(torch.isfinite(v).all() for v in ck['weights'].values()))
    summary=pd.read_csv(out/'tables/field_v4_factorial.csv')
    per=pd.read_csv(out/'tables/field_v4_by_series.csv.gz')
    check('all four environments and four policies',len(summary)==16 and len(per)==16*N)
    metrics=['holding_cost','stockout_cost','total_cost','stockout_ratio']
    expected=per.groupby(['environment','policy'])[metrics].mean().sort_index()
    actual=summary.set_index(['environment','policy']).sort_index()
    np.testing.assert_allclose(actual[metrics],expected,atol=1e-6)
    checks.append('all factorial table means equal per-series outcomes')
    for env,g in per.groupby('environment'):
        baseline=g[g.policy=='Normal'].set_index('series_id')
        for method,x in g.groupby('policy'):
            x=x.set_index('series_id').sort_index();s=actual.loc[env,method]
            np.testing.assert_equal(x.initial_stock.to_numpy(),baseline.initial_stock.to_numpy())
            np.testing.assert_allclose(s.paired_difference,(x.total_cost-baseline.total_cost).mean(),atol=1e-6)
            np.testing.assert_allclose(s.relative_to_Normal_pct,100*(x.total_cost.mean()/baseline.total_cost.mean()-1),atol=1e-6)
        checks.append(env+' identical initial stock and correct paired differences')
    initial=pd.read_csv(out/'tables/field_initial_state.csv').set_index('series_id')
    field=per[per.environment=='lead2_hybrid']
    for method,x in field.groupby('policy'):
        tr=pd.read_csv(out/f'trajectories/v4_field_{method}_daily.csv.gz').sort_values(['series_id','day'])
        orders=pd.read_csv(out/f'decisions/v4_field_{method}_orders.csv.gz')
        check(method+' complete daily observations',len(tr)==30*N and not tr.duplicated(['series_id','day']).any())
        check(method+' finite nonnegative actions',np.isfinite(orders.order_quantity).all() and orders.order_quantity.ge(0).all())
        np.testing.assert_equal((orders.arrival_day-orders.order_day).to_numpy(),2*p['lead'][orders.series_id,orders.order_day])
        np.testing.assert_allclose(tr.holding_cost,12*np.maximum(tr.end_inventory,0),atol=1e-6)
        np.testing.assert_allclose(tr.stockout_cost,88*(np.maximum(-tr.end_inventory,0)+tr.lost_sales),atol=1e-6)
        previous=tr.groupby('series_id').end_inventory.shift();first=tr.groupby('series_id').cumcount()==0
        previous.loc[first]=initial.loc[tr.loc[first,'series_id'],'initial_inventory'].to_numpy()
        np.testing.assert_allclose(tr.end_inventory,previous+tr.received-tr.demand+tr.lost_sales,atol=1e-6)
        old_open=tr.groupby('series_id').pending_after_day.shift()
        old_open.loc[first]=initial.loc[tr.loc[first,'series_id'],'initial_pending'].to_numpy()
        placed=orders.groupby(['series_id','order_day']).order_quantity.sum()
        q=pd.MultiIndex.from_frame(tr[['series_id','day']]);placed=placed.reindex(q,fill_value=0).to_numpy()
        np.testing.assert_allclose(tr.pending_after_day,old_open+placed-tr.received,atol=1e-6)
        totals=tr.groupby('series_id')[['holding_cost','stockout_cost']].sum().sort_index()
        x=x.set_index('series_id').sort_index()
        np.testing.assert_allclose(totals,x[['holding_cost','stockout_cost']],atol=1e-6)
        checks.append(method+' inventory and open-order conservation, losses charged once, costs sum')
    pair=pd.read_csv(out/'tables/field_matched_pairs.csv')
    treatment=field[field.policy=='V4_MatchedCalibrator'].set_index('series_id').loc[pair.treated_id]
    control=field[field.policy=='Normal'].set_index('series_id').loc[pair.control_id]
    t2=pd.read_csv(out/'tables/table2_synthetic_field.csv').set_index('metric')
    for metric in ['holding_cost','stockout_cost','total_cost','turnover_days','stockout_ratio']:
        np.testing.assert_allclose(t2.loc[metric,'treatment_mean'],treatment[metric].mean(),atol=1e-6)
        np.testing.assert_allclose(t2.loc[metric,'control_mean'],control[metric].mean(),atol=1e-6)
    checks.append('PSM table outcomes come from same logged policies')
    holdout=json.loads((out/'logs/independent_holdout_validation.json').read_text('utf-8'))
    check('fixed independent panels and weights unchanged',holdout['passed'] and holdout['weights_unchanged'] and holdout['n_panels']==3)
    check('holdout declared evaluation-only',holdout['not_used_for_selection'])
    for name,h in holdout['model_sha256'].items():check('evaluation immutable '+name,sha(out/'models'/name)==h)
    hs=pd.read_csv(out/'tables/independent_holdout_summary.csv')
    hp=pd.read_csv(out/'tables/independent_holdout_by_series.csv.gz')
    check('all independent seeds reported',set(hs.panel_seed)=={20261020,20261021,20261022} and len(hs)==12)
    np.testing.assert_allclose(hs.set_index(['panel_seed','policy']).sort_index()[metrics],hp.groupby(['panel_seed','policy'])[metrics].mean().sort_index(),atol=1e-6)
    checks.append('independent-panel summaries equal raw outcomes')
    protocol=json.loads((out/'logs/field_v4_protocol.json').read_text('utf-8'))
    check('no closed-form labels claimed in field',not protocol['field_calibration_uses_closed_form_labels'])
    check('checkpoint not selected on test',not protocol['test_used_for_selection'])
    check('independent simulators agree within float32 bound',protocol['numpy_torch_max_cost_difference']<1.)
    result=dict(passed=True,n_checks=len(checks),checks=checks,profile=profile,source='SYNTHETIC V4 engineering extension')
    (out/'logs/field_v4_output_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'profile':profile}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='standard');verify(ap.parse_args().profile)
