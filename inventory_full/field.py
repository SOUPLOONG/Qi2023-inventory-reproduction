"""Synthetic field intervention: real policy rollouts, pre-treatment PSM, clustered inference."""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
from .engine import plan,simulate
from .training import predict
from .core import csv,dump

METRICS=['holding_cost','stockout_cost','total_cost','turnover_days','stockout_ratio']

def run(p,cfg,out,model,treatment_policy=None,post_plan=None,treatment_label='V3_Transfer'):
    N=len(p['scale']);pre_start=330;post_start=370;days=30
    # Continuous policy history: 30-day burn-in, pre, 10-day bridge, then intervention.
    # At every boundary inventory AND outstanding arrivals are carried forward.
    state={}
    burnP=plan(p,300,30,cfg,field=True,anchor=300)
    simulate(p,burnP,cfg,300,30,'Normal',h=12,b=88,field=True,final_state=state)
    preP=plan(p,pre_start,days,cfg,field=True,anchor=300);postP=plan(p,post_start,days,cfg,field=True,anchor=300)
    epred=predict(model,postP['x'],cfg)
    if post_plan is not None:postP=post_plan
    after_pre={};pre,*_=simulate(p,preP,cfg,pre_start,days,'Normal',h=12,b=88,field=True,initial_state=state,final_state=after_pre)
    bridgeP=plan(p,360,10,cfg,field=True,anchor=300);at_intervention={}
    simulate(p,bridgeP,cfg,360,10,'Normal',h=12,b=88,field=True,initial_state=after_pre,final_state=at_intervention)
    postC,*_=simulate(p,postP,cfg,post_start,days,'Normal',h=12,b=88,field=True,initial_state=at_intervention)
    method='E2E_calibrated' if treatment_policy is not None else 'E2E_RNN'
    postT,trace,orders,_=simulate(p,postP,cfg,post_start,days,method,model=model,pred=epred,h=12,b=88,field=True,keep=True,initial_state=at_intervention,policy=treatment_policy)
    csv(trace,out/'trajectories/field_E2E_daily.csv.gz')
    csv(pd.DataFrame({'series_id':np.arange(N),'initial_inventory':at_intervention['inventory'],
        'initial_pending':at_intervention['pending'].sum(1)}),out/'tables/field_initial_state.csv')
    # Recalibrate action cost ratio: field must use a separately retrained b/h policy.
    # Caller supplies field model retrained for 88/12; passed costs use exact units12/88.
    mu=p['demand'][:,pre_start-180:pre_start].mean(1)
    lm=p['lead'][:,pre_start-14:pre_start-8].mean(1)*2
    cov=np.c_[np.log1p(mu),lm]
    centered=(cov-cov.mean(0))/np.maximum(cov.std(0),1e-6)
    probability=expit(.6*centered[:,0]-.45*centered[:,1])
    rng=np.random.default_rng(cfg['seed']+71);treated=rng.random(N)<probability
    propensity=LogisticRegression(max_iter=1000,random_state=cfg['seed']).fit(cov,treated)
    score=propensity.predict_proba(cov)[:,1]
    ti=np.flatnonzero(treated);ci=np.flatnonzero(~treated)
    match=NearestNeighbors(n_neighbors=1).fit(score[ci,None]).kneighbors(score[ti,None],return_distance=False).ravel()
    matched=ci[match]
    csv(pd.DataFrame({'treated_id':ti,'control_id':matched,'treated_score':score[ti],'control_score':score[matched]}),out/'tables/field_matched_pairs.csv')
    csv(pd.DataFrame({'series_id':np.arange(N),'treatment':treated.astype(int),'true_assignment_probability':probability,
        'estimated_score':score,'pre_demand':mu,'pre_lead':lm}),out/'tables/field_assignment.csv')
    # Duplicate controls receive their actual matching multiplicity; clusters use original series ID.
    selected=np.r_[ti,matched];treat=np.r_[np.ones(len(ti)),np.zeros(len(ti))]
    pre_selected=pre.iloc[selected].copy().reset_index(drop=True)
    post_selected=pd.concat([postT.iloc[ti],postC.iloc[matched]],ignore_index=True)
    all_rows=[];table2=[];table3=[];table4=[]
    for period,df in [(0,pre_selected),(1,post_selected)]:
        a=df.copy();a['treatment']=treat;a['post']=period;a['cluster_id']=selected;a['pre_demand']=mu[selected];a['pre_lead']=lm[selected]
        all_rows.append(a)
    panel=pd.concat(all_rows,ignore_index=True)
    for metric in METRICS:
        yt=postT.iloc[ti][metric].to_numpy();yc=postC.iloc[matched][metric].to_numpy()
        X=sm.add_constant(pd.DataFrame({'treated':treat}))
        fit=sm.OLS(np.r_[yt,yc],X).fit(cov_type='cluster',cov_kwds={'groups':selected})
        table2.append({'metric':metric,'treatment_mean':yt.mean(),'control_mean':yc.mean(),
            'difference':yt.mean()-yc.mean(),'relative_change_pct':100*(yt.mean()/max(yc.mean(),1e-12)-1),
            'p_clustered':fit.pvalues['treated'],'p_welch_descriptive':ttest_ind(yt,yc,equal_var=False).pvalue,
            'p_student_descriptive':ttest_ind(yt,yc,equal_var=True).pvalue,
            'matched_pairs':len(ti)})
        X=sm.add_constant(pd.DataFrame({'treated':treat,'pre_demand':mu[selected],'pre_lead':lm[selected]}))
        fit=sm.OLS(post_selected[metric].to_numpy(),X).fit(cov_type='cluster',cov_kwds={'groups':selected})
        for term in X.columns:table3.append({'metric':metric,'term':term,'coefficient':fit.params[term],
            'standard_error_clustered':fit.bse[term],'p_clustered':fit.pvalues[term]})
        X=sm.add_constant(panel[['treatment','post']]);X['treatment_post']=panel.treatment*panel.post
        fit=sm.OLS(panel[metric],X).fit(cov_type='cluster',cov_kwds={'groups':panel.cluster_id})
        means=panel.groupby(['treatment','post'])[metric].mean()
        table4.append({'metric':metric,'treatment_pre':means[1,0],'treatment_post':means[1,1],
            'control_pre':means[0,0],'control_post':means[0,1],'did':fit.params['treatment_post'],
            'standard_error_clustered':fit.bse['treatment_post'],'p_clustered':fit.pvalues['treatment_post']})
    csv(pd.DataFrame(table2),out/'tables/table2_synthetic_field.csv');csv(pd.DataFrame(table3),out/'tables/table3_synthetic_regression.csv')
    csv(pd.DataFrame(table4),out/'tables/table4_synthetic_did.csv');csv(panel,out/'tables/synthetic_field_panel.csv.gz')
    crossing=0
    for _,g in orders.groupby('series_id'):
        crossing+=int((np.diff(g.sort_values('order_day').arrival_day.to_numpy())<0).sum())
    csv(orders,out/'decisions/field_E2E_orders.csv.gz')
    balance=[]
    for j,name in enumerate(['log_pre_demand','pre_lead']):
        den=np.sqrt((cov[ti,j].var()+cov[ci,j].var())/2)
        balance.append({'variable':name,'SMD_before':(cov[ti,j].mean()-cov[ci,j].mean())/den,
                        'SMD_after':(cov[ti,j].mean()-cov[matched,j].mean())/den})
    csv(pd.DataFrame(balance),out/'tables/field_covariate_balance.csv')
    dump(out/'logs/field_protocol.json',{'source':'SYNTHETIC policy intervention, not JD field experiment',
        'n_series':N,'n_treated':len(ti),'matched_pairs':len(ti),'unique_matched_controls':len(np.unique(matched)),
        'pre':[pre_start,pre_start+days-1],'post':[post_start,post_start+days-1],'h':12,'b':88,
        'continuous_history':'burn-in300..329; pre330..359; bridge360..369; intervention370..399. Inventory and pending orders carried across all boundaries; schedules anchored day300.',
        'treatment_policy':treatment_label,
        'training_scope':('V4 extension: field-cost-trained calibration, 2..8-day leads and hybrid shortages matched in train/validation/test' if treatment_policy is not None else 'cost ratio retrained on offline no-crossover backorder labels; 2..8-day leads and hybrid losses are out-of-distribution stress tests'),
        'observation':'true historical demand fully observed, including unmet/lost demand',
        'field_model_retrained_cost_ratio':88/12,'with_replacement':True,'matching':'estimated propensity score, pre-treatment covariates only',
        'cluster':'original synthetic series ID; repeated matches and both periods clustered together',
        'supply':'offline vendor lead doubled to 2..8 days; order arrivals can cross','crossings_in_E2E_orders':crossing,
        'shortage':'hybrid backlog capped by pending orders; uncovered demand is lost and penalized once',
        'turnover':'average positive on-hand / average demand, days, lower means less inventory occupation',
        'identification':'Simulated assignment uses observed pre-covariates; no claim of real-world causal validation'})
    return score[ti],score[ci],score[matched]
