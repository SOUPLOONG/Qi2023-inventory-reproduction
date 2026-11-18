"""Validate exported results independently of plot/report rendering."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

def verify(root,profile):
    out=root/'results'/profile;data=root/'data'/profile;checks=[]
    cfg=json.loads((out/'config_used.json').read_text('utf-8'));N=cfg['n_series'];T=cfg['test_days']
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    panel=np.load(data/'panel.npz');summary=pd.read_csv(out/'tables/costs_main.csv')
    # Simulator initializes a float32 state. Pandas writes that column with
    # float32's shortest round-trip decimal; recover its original bits before
    # comparing against float64 daily traces, rather than relaxing conservation.
    initial_expected=(panel['demand'][:,cfg['test_start']-180:cfg['test_start']].mean(1)*panel['review']/2).astype('float64')
    per=pd.read_csv(out/'tables/costs_by_series_main.csv')
    check('seven genuine methods',len(summary)==7)
    check('all series in every method',per.groupby('method').size().eq(N).all())
    init=[]
    for method in summary.method:
        tr=pd.read_csv(out/f'trajectories/{method}_daily.csv.gz')
        orders=pd.read_csv(out/f'decisions/{method}_orders.csv.gz')
        check(method+' complete daily trace',len(tr)==N*T and not tr.duplicated(['series_id','day']).any())
        check(method+' nonnegative orders',(orders.order_quantity>=0).all())
        check(method+' actual lead',np.array_equal(orders.arrival_day.to_numpy()-orders.order_day.to_numpy(),panel['lead'][orders.series_id,orders.order_day]))
        np.testing.assert_allclose(tr.holding_cost,np.maximum(tr.end_inventory,0),atol=1e-6)
        np.testing.assert_allclose(tr.stockout_cost,9*np.maximum(-tr.end_inventory,0),atol=1e-6)
        daily=tr.groupby('series_id')[['holding_cost','stockout_cost']].sum().sort_index()
        block=per[per.method==method].set_index('series_id').sort_index()
        np.testing.assert_allclose(daily,block[['holding_cost','stockout_cost']],atol=1e-6)
        np.testing.assert_allclose(daily.sum(axis=1),block.total_cost,atol=1e-6)
        g=tr.sort_values(['series_id','day']).copy()
        previous=g.groupby('series_id').end_inventory.shift()
        first=g.groupby('series_id').cumcount()==0
        np.testing.assert_equal(block.initial_stock.to_numpy(dtype='float32'),initial_expected.astype('float32'))
        previous.loc[first]=initial_expected[g.loc[first,'series_id'].to_numpy()]
        np.testing.assert_allclose(g.end_inventory,previous+g.received-g.demand,atol=1e-6)
        init.append(block.initial_stock.to_numpy())
        checks.append(method+' cost sums and inventory mass balance')
    for i in init[1:]:np.testing.assert_equal(init[0],i)
    checks.append('identical initial stock across methods')
    keys=['holding_cost','stockout_cost','total_cost','turnover_days','stockout_ratio']
    aggregate=per.groupby('method')[keys].mean().sort_index()
    actual=summary.set_index('method').sort_index()
    np.testing.assert_allclose(actual[keys],aggregate,atol=1e-6)
    np.testing.assert_allclose(actual.relative_gap_pct,(actual.total_cost/actual.loc['OPT','total_cost']-1)*100,atol=1e-6)
    checks.append('Figure2 summary table and OPT gap equal underlying per-series means')
    wide=per.pivot(index='series_id',columns='method',values='total_cost')
    check('oracle lower bound on every series',(wide.drop(columns='OPT').sub(wide.OPT,axis=0)>=-1e-3).all().all())
    cover=json.loads((out/'logs/coverage_manifest.json').read_text('utf-8'))
    check('every main/appendix table present',cover['all_expected_tables_present'])
    check('main/appendix figure coverage',all((out/f'figures/{s}.png').exists() for s in ['figure1_architecture','figure2','figure3_matching','figureB1_data_size','figureB2_cost_ratio','figureB3_initialization']))
    expected=22 if (out/'logs/field_v4_protocol.json').exists() else 20
    check('expected neural checkpoints',len(list((out/'models').glob('*.pt')))==expected)
    actual_hash=hashlib.sha256((data/'training_samples.npz').read_bytes()).hexdigest()
    for path in (out/'logs').glob('model_*.json'):
        meta=json.loads(path.read_text('utf-8'))
        check(path.stem+' dataset identity',meta['spec']['data_signature']==actual_hash)
        check(path.stem+' declared epoch budget',meta['spec']['epochs']==cfg['epochs'])
    pairs=pd.read_csv(out/'tables/field_matched_pairs.csv');assignment=pd.read_csv(out/'tables/field_assignment.csv').set_index('series_id')
    check('PSM treatment IDs actual assignment',assignment.loc[pairs.treated_id,'treatment'].eq(1).all())
    check('PSM control IDs actual assignment',assignment.loc[pairs.control_id,'treatment'].eq(0).all())
    fp=pd.read_csv(out/'tables/synthetic_field_panel.csv.gz')
    check('field stable clusters',fp.cluster_id.nunique()<=N and len(fp)==len(pairs)*4)
    did=pd.read_csv(out/'tables/table4_synthetic_did.csv')
    np.testing.assert_allclose(did.did,(did.treatment_post-did.treatment_pre)-(did.control_post-did.control_pre),atol=1e-6)
    checks.append('DID coefficients equal direct four-cell differences')
    for table in ['table2_synthetic_field.csv','table3_synthetic_regression.csv','table4_synthetic_did.csv']:
        df=pd.read_csv(out/'tables'/table);check(table+' finite inference',np.isfinite(df.select_dtypes('number')).all().all())
    result={'passed':True,'checks':checks,'n_checks':len(checks),'profile':profile,'daily_rows':N*cfg['days'],
            'scope':'synthetic reconstructed methods; not original JD numerical replication'}
    (out/'logs/output_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'checks':len(checks),'profile':profile}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='standard');args=ap.parse_args()
    verify(Path(__file__).resolve().parent,args.profile)
