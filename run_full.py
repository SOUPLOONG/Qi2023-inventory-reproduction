"""Public entry point: python run_full.py --profile standard --stage all"""
from pathlib import Path
import sys,os,argparse,json,time,platform,importlib.metadata,warnings

ROOT=Path(__file__).resolve().parent
if os.name=='nt' and (ROOT/'.vendor').exists():sys.path.insert(0,str(ROOT/'.vendor'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['quick','standard','large'],default='standard')
    parser.add_argument('--stage',choices=['all','generate','main','sensitivity','field','author','figures','report'],default='all')
    parser.add_argument('--device',default='auto');parser.add_argument('--epochs',type=int)
    parser.add_argument('--optimizer',choices=['adam','sgd']);parser.add_argument('--batch-size',type=int)
    parser.add_argument('--field-epochs',type=int)
    parser.add_argument('--skip-holdout',action='store_true',help='Skip independent synthetic-panel evaluation, without changing training')
    parser.add_argument('--output',type=Path);parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/f'configs/{args.profile}.json').read_text('utf-8'))
    if args.stage=='report':
        out=(args.output or ROOT/'results'/args.profile).resolve()
        from inventory_full.report import report
        if not (out/'logs/session.json').exists():parser.error('No completed session; run experiments first')
        previous=json.loads((out/'logs/session.json').read_text('utf-8'))
        used=json.loads((out/'config_used.json').read_text('utf-8'))
        report(ROOT,out,used,previous)
        print('Report rebuilt from recorded run: '+str(out/'复现报告.html'));return
    if args.epochs is not None:cfg['epochs']=args.epochs
    if args.optimizer is not None:cfg['optimizer']=args.optimizer
    if args.batch_size is not None:cfg['batch_size']=args.batch_size
    cfg['field_epochs']=args.field_epochs if args.field_epochs is not None else (4 if args.profile=='quick' else 40 if args.profile=='large' else 20)
    if cfg['field_epochs']<1:parser.error('field-epochs must be positive')
    if cfg['epochs']<1 or cfg['batch_size']<1:parser.error('epochs and batch-size must be positive')
    if (cfg['val_start'],cfg['test_start'])!=(280,330):
        parser.error('This dataset builder uses fixed purged split boundaries 280/330; modify builder and audit together for other splits')
    import torch
    torch.set_num_threads(4)
    cfg['device']=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    cfg['force']=args.force
    out=(args.output or ROOT/'results'/args.profile).resolve()
    for d in ['tables','logs','models','figures','trajectories','decisions']:(out/d).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'logs/matplotlib_cache'))
    from inventory_full.core import dump,digest,csv
    from inventory_full.data import generate,build_samples
    import numpy as np,pandas as pd
    start=time.perf_counter();data=ROOT/'data'/args.profile
    if (data/'panel.npz').exists() and not args.force:
        prov=json.loads((data/'provenance.json').read_text('utf-8'))
        if (prov['n_series'],prov['seed'],prov['days'])!=(cfg['n_series'],cfg['seed'],cfg['days']):raise ValueError('Data configuration differs; use a different profile or --force')
        if digest(data/'panel.npz')!=prov['panel_sha256']:raise ValueError('Panel checksum differs from provenance')
        with np.load(data/'panel.npz') as z:p={k:z[k] for k in z.files}
    else:
        print('[data] generating continuous synthetic panel',flush=True);p=generate(data,cfg)
    cache_spec={'panel_sha256':digest(data/'panel.npz'),'horizon':cfg['horizon'],
        'construction_source':[digest(ROOT/'inventory_full'/f) for f in ['data.py','core.py']]}
    if (data/'training_samples.npz').exists() and not args.force:
        manifest=data/'sample_configuration.json'
        if not manifest.exists() or json.loads(manifest.read_text('utf-8'))!=cache_spec:
            raise ValueError('Training-data cache signature differs or is missing; use --force or a separate dataset profile')
        with np.load(data/'training_samples.npz') as z:ds={k:z[k] for k in z.files}
    else:
        print('[data] building causal states and matured labels',flush=True);ds=build_samples(p,cfg,data)
        dump(data/'sample_configuration.json',cache_spec)
    cfg['data_signature']=digest(data/'training_samples.npz')
    dump(out/'config_used.json',cfg)
    csv(pd.DataFrame({'category':np.arange(6),'n_series':np.bincount(p['category'],minlength=6)}),out/'tables/table1_synthetic_composition.csv')
    params=[('MQRNN hidden size',50,50),('VLT layers','50,20','50,20'),('Fusion layers','100,100','100,100'),
            ('Learning rate',.001,.001),('Batch size',64,cfg['batch_size']),('Momentum',.85,'SGD=.85; Adam profile used by default'),
            ('Dropout',.2,.2),('Optimizer','SGD family discussed; details incomplete',cfg['optimizer']),('Epochs','not reported',cfg['epochs']),
            ('Quantiles','text has count ambiguity','.1,.5,.6,.7,.8,.9,.95 + mean'),('Forecast horizon','not reported',cfg['horizon'])]
    csv(pd.DataFrame(params,columns=['parameter','paper_reference','implementation']),out/'tables/table_B1_hyperparameters.csv')
    completed=[];captured=[]
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always')
        model=forecast=tree=base=selected=meta=None
        if args.stage in ['all','main','sensitivity','field','figures']:
            from inventory_full.experiments import main as main_exp
            model,forecast,tree,base,selected,meta=main_exp(p,ds,cfg,out)
            from inventory_full.figures import main_figures
            main_figures(out,cfg);completed.append('main')
        if args.stage in ['all','sensitivity']:
            from inventory_full.experiments import sensitivity
            sensitivity(p,ds,cfg,out,model,forecast,tree,base,selected,meta);completed.append('sensitivity')
        if args.stage in ['all','field']:
            from inventory_full.training import fit
            from inventory_full.field_calibration import run
            field_model,_=fit(ds,cfg,out,'field_ratio',b=88/12)
            run(p,cfg,out,model,field_model);completed.append('field')
            if not args.skip_holdout:
                from inventory_full.field_holdout import run as holdout
                holdout(cfg,out);completed.append('independent_holdout')
        if args.stage in ['all','author']:
            if (ROOT/'data/author/df_test.csv').exists():
                from author_reference.qi_repro.offline import run as offline
                from author_reference.qi_repro.field import run as field,matching as author_matching
                aout=out/'author_disclosed';offline(ROOT/'data/author',aout);field(ROOT/'data/author',aout)
                author_matching(ROOT/'data/author',aout)
                completed.append('author_disclosed')
        captured=list(dict.fromkeys(str(w.message) for w in ww))
    session={'profile':args.profile,'stage':args.stage,'completed':completed,'elapsed_seconds':time.perf_counter()-start,
        'python':sys.version,'platform':platform.platform(),'device':cfg['device'],
        'gpu':torch.cuda.get_device_name(0) if cfg['device'].startswith('cuda') else None,
        'versions':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','torch','lightgbm','statsmodels','scikit-learn','matplotlib']},
        'warnings':captured,'source':'SYNTHETIC full-scope reconstruction + separate disclosed-data reproduction',
        'daily_rows':int(cfg['n_series']*cfg['days']),'training_samples':int((ds['split']=='train').sum()),
        'validation_samples':int((ds['split']=='val').sum())}
    dump(out/'logs/session.json',session)
    from inventory_full.report import report
    report(ROOT,out,cfg,session)
    print(f'Finished {completed} in {session["elapsed_seconds"]:.1f}s. Report: {out / "复现报告.html"}',flush=True)

if __name__=='__main__':main()
