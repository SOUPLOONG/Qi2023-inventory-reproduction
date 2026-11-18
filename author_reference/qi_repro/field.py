"""现场公开子样本复算；不把500行子样本称为正式版全样本。"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from .common import numeric_list,save_csv,dump_json
from .inventory import inventory_cost

METRICS=['holding_cost','stockout_cost','total_cost','turnover_days','stockout_ratio']


def row_metrics(frame,h,b):
    records=[]
    for _,r in frame.iterrows():
        inv=numeric_list(r.test_inv)
        # 忠实原Notebook：最后一个库存元素不计费，但缺货率分母仍为len(inv)。
        scored=inv[:-1]
        turn=float(r.ave_inv/r.ave_demand)
        records.append({'id':r.sku_dc_pair_index,**inventory_cost(scored,h,b),
            'turnover_raw':turn,'turnover_days':max(0.0,turn),
            'stockout_ratio':float((scored<0).sum()/len(inv)),
            'n_inventory_points':len(inv),'n_scored_points':len(scored),
            'ave_demand':float(r.ave_demand),'vlt':float(str(r.vlt).split()[0])})
    return pd.DataFrame(records)


def run(data_root,out):
    source={(group,period):pd.read_csv(data_root/f'order_{group}_{period}_500.csv')
            for group in ['e2e','jd'] for period in ['pre','post']}
    all_summaries=[];all_tests=[];all_regs=[];all_did=[];all_rows=[]
    for spec,h,b,intercept in [('author_checkpoint',1,9,False),('paper_hb_intercept_sensitivity',12,88,True)]:
        blocks={k:row_metrics(df,h,b) for k,df in source.items()}
        for (group,period),df in blocks.items():
            row={'specification':spec,'group':group,'period':period,'n':len(df),'h':h,'b':b}
            row.update(df[METRICS].mean().to_dict());all_summaries.append(row)
            all_rows.append(df.assign(specification=spec,group=group,period=period))
        t,c=blocks['e2e','post'],blocks['jd','post']
        X=pd.DataFrame({'e2e':np.r_[np.ones(len(t)),np.zeros(len(c))],
                        'demand':pd.concat([t,c]).ave_demand.to_numpy(),
                        'vlt':pd.concat([t,c]).vlt.to_numpy()})
        if intercept:X=sm.add_constant(X,has_constant='add')
        for metric in METRICS:
            a=t[metric].to_numpy();bvals=c[metric].to_numpy()
            test=stats.ttest_ind(a,bvals,equal_var=True)
            all_tests.append({'specification':spec,'metric':metric,'treatment_mean':a.mean(),
                'control_mean':bvals.mean(),'difference':a.mean()-bvals.mean(),
                't':float(test.statistic),'p':float(test.pvalue)})
            fit=sm.OLS(np.r_[a,bvals],X).fit()
            for term in fit.params.index:
                all_regs.append({'specification':spec,'metric':metric,'term':term,
                    'coefficient':fit.params[term],'standard_error':fit.bse[term],
                    'p':fit.pvalues[term],'n':int(fit.nobs),'has_intercept':intercept})
            # 原包DiD的t检验基于文件行位置的组间差，不假定ID已确认可链接。
            # 原包post组jd turnover未截断，DiD此项也原样保留。
            post_control_key='turnover_raw' if metric=='turnover_days' else metric
            post=t[metric].to_numpy()-c[post_control_key].to_numpy()
            pre=blocks['e2e','pre'][metric].to_numpy()-blocks['jd','pre'][metric].to_numpy()
            test=stats.ttest_ind(post,pre,equal_var=True)
            all_did.append({'specification':spec,'metric':metric,
                'treatment_pre':blocks['e2e','pre'][metric].mean(),
                'treatment_post':t[metric].mean(),
                'control_pre':blocks['jd','pre'][metric].mean(),
                'control_post':c[post_control_key].mean(),
                'did':float(post.mean()-pre.mean()),'legacy_rowwise_t':float(test.statistic),
                'legacy_rowwise_p':float(test.pvalue)})
    for name,rows in [('field_summary',all_summaries),('field_ttests',all_tests),
                      ('field_regressions',all_regs),('field_did',all_did)]:
        save_csv(pd.DataFrame(rows),out/f'tables/{name}.csv')
    save_csv(pd.concat(all_rows,ignore_index=True),out/'tables/field_row_metrics.csv')
    dump_json(out/'logs/field_scope.json',{
        'scope':'four disclosed tables, each 500 rows; not full article sample',
        'author_checkpoint':'h=1,b=9; no intercept; sum inv[:-1]; ratio denominator len(inv)',
        'paper_hb_intercept_sensitivity':'only changes h,b to 12,88 and adds intercept; same disclosed windows and scoring convention',
        'did_warning':'Legacy rowwise t-test retained for code comparison. Stable cross-file panel linkage is not verified. No causal guarantee.',
        'normalization':'Costs are sums per record, then unweighted means. Unequal window lengths are retained, not silently standardized.',
        'turnover':'ave_inv/ave_demand; negative values clipped to zero as in notebook except legacy post-control DiD vector.'})
    matching(data_root,out)
    return pd.DataFrame(all_summaries)


def matching(data_root,out):
    treated=pd.read_csv(data_root/'order_e2e_post_500.csv')[['ave_demand','vlt_num']].copy()
    control=pd.read_csv(data_root/'control_demand_vlt.csv')[['ave_demand','vlt_num']].copy()
    treated['e2e']=1;control['e2e']=0
    both=pd.concat([treated,control],ignore_index=True)
    model=LogisticRegression(max_iter=100).fit(both[['ave_demand','vlt_num']],both.e2e)
    scores=model.predict_proba(both[['ave_demand','vlt_num']])[:,1]
    # 原包实际在原始特征（包括组别常数列）上做有放回的最近邻匹配。
    nn=NearestNeighbors(n_neighbors=1,algorithm='auto').fit(control)
    dist,idx=nn.kneighbors(treated);idx=idx.ravel()
    matched=control.iloc[idx]
    save_csv(pd.DataFrame({'treated_row':np.arange(len(treated)),'candidate_row':idx,
        'distance_raw_features':dist.ravel(),'treated_score':scores[:len(treated)],
        'matched_score':scores[len(treated)+idx]}),out/'tables/matching_indices.csv')
    rows=[]
    for col in ['ave_demand','vlt_num']:
        denom=np.sqrt((treated[col].var(ddof=1)+control[col].var(ddof=1))/2)
        rows.append({'variable':col,'treated_mean':treated[col].mean(),'candidate_mean':control[col].mean(),
            'matched_mean':matched[col].mean(),'smd_before':(treated[col].mean()-control[col].mean())/denom,
            'smd_after_same_denominator':(treated[col].mean()-matched[col].mean())/denom})
    save_csv(pd.DataFrame(rows),out/'tables/matching_balance.csv')
    dump_json(out/'logs/matching_scope.json',{'n_treated':len(treated),'n_candidates':len(control),
        'unique_matched_candidates':int(len(np.unique(idx))),'with_replacement':True,
        'distance':'raw features, NOT fitted propensity-score distance',
        'outcomes':'No outcome reconstruction for newly matched candidates; only matching covariates are disclosed.'})
    from .plots import matching_chart
    matching_chart(scores[:len(treated)],scores[len(treated):],scores[len(treated)+idx],out/'figures/matching_scores.png')
