import numpy as np
import pandas as pd
from .training import fit,fit_tree
from .engine import evaluate_model
from .core import csv,dump

def main(p,ds,cfg,out):
    model,meta=fit(ds,cfg,out,'main')
    forecast,fmeta=fit(ds,cfg,out,'forecast',forecast_only=True)
    tree=fit_tree(ds,out,cfg)
    summary,selected=evaluate_model(p,cfg,out,model,forecast,tree,keep=True)
    dump(out/'logs/baseline_selection.json',selected)
    return model,forecast,tree,summary,selected,meta

def sensitivity(p,ds,cfg,out,model,forecast,tree,base,selected,mainmeta):
    all_tables={};weights=[]
    settings={'layers':[1,2,3,4],'hidden':[30,40,50,60],'lr':[.0001,.001,.01],
              'fraction':[.2,.4,.6,.8,1.],'ratio':[1.,3.,5.,9.,19.],
              'seed':[cfg['seed'],cfg['seed']+1]}
    defaults={'layers':2,'hidden':50,'lr':.001,'fraction':1.,'ratio':9.,'seed':cfg['seed']}
    for kind,values in settings.items():
        rows=[]
        for value in values:
            print(f'[sensitivity] {kind} = {value}',flush=True)
            if value==defaults[kind]:m,meta,summary=model,mainmeta,base
            else:
                kwargs={kind:value} if kind!='ratio' else {'b':value}
                tag=f'{kind}_{value:g}'
                m,meta=fit(ds,cfg,out,tag,**kwargs)
                summary,_=evaluate_model(p,cfg,out,m,forecast,tree,b=value if kind=='ratio' else 9.,
                                         tag=tag,keep=False,selected=selected)
            if kind=='seed':weights.append((int(value),m.vlt[0].weight.detach().cpu().numpy()))
            if kind=='ratio':
                for _,r in summary.iterrows():
                    if r.method in ['OPT','E2E_RNN','Normal','Gamma']:
                        rows.append({'value':value,**r.to_dict(),'train_seconds':meta['train_seconds'],'training_series':meta['training_series']})
            else:
                r=summary[summary.method=='E2E_RNN'].iloc[0]
                rows.append({'value':value,**r.to_dict(),'train_seconds':meta['train_seconds'],
                    'training_series':meta['training_series'],'training_samples':meta['train_samples'],
                    'best_epoch':meta['best_epoch']})
        table=pd.DataFrame(rows);all_tables[kind]=table
        label={'layers':'B2_layers','hidden':'B3_neurons','lr':'B4_learning_rate','fraction':'B5_data_size','ratio':'cost_ratio','seed':'B6_initialization'}[kind]
        csv(table,out/f'tables/table_{label}.csv')
    np.savez_compressed(out/'models/initialization_weights.npz',**{f'seed_{seed}':w for seed,w in weights})
    from .figures import appendix
    appendix(all_tables,out,weights)
    dump(out/'logs/sensitivity_protocol.json',{'all_nondefault_settings_retrained':True,
        'cost_ratio_changes_labels_and_actions':True,'settings':settings,
        'data_fraction':'fraction of unique training series; same validation and test panels',
        'selection':'best epoch by validation loss; no choice by test ranking',
        'budgets':'same epoch count and batch size across settings; measured time may vary with hardware',
        'source':'SYNTHETIC, no expectation of matching original numerical values or rankings'})
    return all_tables
