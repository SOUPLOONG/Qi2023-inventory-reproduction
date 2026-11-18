"""忠实重算披露包E2E_Section4检查点；绝不把简化模拟器称为式(1)。"""
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm
from .common import numeric_list, save_csv, dump_json
from .inventory import inventory_cost

METHODS=['OPT','E2E_RNN','Bm1','Bm2','Normal','Gamma','gbm']


def load_test(data_root):
    frame=pd.read_csv(data_root/'df_test.csv')
    frame['sf_rnn']=frame.sf_rnn.map(numeric_list)
    frame['demand_hist']=frame.demand_hist.map(lambda s:numeric_list(s, comma=False))
    return frame


def add_benchmarks(frame):
    df=frame.copy()
    mean_forecast=df.sf_rnn.map(np.mean)
    df['Bm1_pred']=mean_forecast*(df.review_period+df.vlt).astype(int)
    def days(rl):
        return int(rl) if rl<=9 else int(rl)-rl//10
    df['Bm2_pred']=mean_forecast*(df.review_period+df.vlt).map(days)
    df['Normal_pred']=(df.demand_mean*(df.review_period+df.vendor_vlt_mean)
        +1.64*np.sqrt((df.review_period+df.vendor_vlt_mean)*df.demand_std**2
                       +df.demand_std**2*df.vendor_vlt_std)).astype(int)
    def gamma_base(row):
        theta=row.demand_std**2/(row.demand_mean+1e-4)
        k=row.demand_mean/(theta+1e-4)
        val=gamma.ppf(.9, a=int(row.review_period+row.vendor_vlt_mean)*k, scale=theta)
        return 0 if np.isnan(val) else int(val)
    df['Gamma_pred']=df.apply(gamma_base,axis=1)
    # 仅修改式(7)的分位数和方差；仍使用原包轨迹，不称为完整正文复现。
    df['Normal_formula7_pred']=np.floor(df.demand_mean*(df.review_period+df.vendor_vlt_mean)
        +norm.ppf(.9)*np.sqrt((df.review_period+df.vendor_vlt_mean)*df.demand_std**2
                              +df.demand_mean**2*df.vendor_vlt_std**2)).astype(int)
    return df


def author_trajectory(group, method):
    """保留原函数：多样本SKU按CSV行序、预测列减库存、负库存到货时重置。

    返回full和scored两条轨迹；原包只对scored计算成本。
    diagnostics显式记录负补货、备用索引、跳过时段，便于理解口径差异。
    """
    stock=[float(group.initial_stock.iloc[0])]; scored=[]
    diag=dict(negative_orders=0, fallback_indices=0, skipped_prearrival_steps=0)
    size=len(group)
    for r,(_,row) in enumerate(group.iterrows()):
        n=len(row.demand_hist)-1 if r<size-1 else len(row.demand_hist)
        for t in range(n):
            if t==0:
                if r==0: base=stock[0]
                else:
                    try:base=stock[-int(round(row.vlt))-1]
                    except IndexError:
                        base=stock[1];diag['fallback_indices']+=1
                order=int(round(row[method+'_pred']-base))
                diag['negative_orders']+=int(order<0)
            lead=int(round(row.vlt))
            if t<lead:
                if r==0:stock.append(stock[-1]-row.demand_hist[t])
                else:diag['skipped_prearrival_steps']+=1
            elif t==lead:
                value=(stock[-1] if stock[-1]>=0 else 0)+order-row.demand_hist[t]
                stock.append(value);scored.append(value)
            else:
                value=stock[-1]-row.demand_hist[t]
                stock.append(value);scored.append(value)
    return stock[1:],scored,diag


def run(data_root,out):
    df=add_benchmarks(load_test(data_root))
    results=[]; diagnostics={m:dict(negative_orders=0,fallback_indices=0,skipped_prearrival_steps=0) for m in METHODS+['Normal_formula7']}
    for sku,group in df.groupby('SKU',sort=True):
        for method in diagnostics:
            full,scored,diag=author_trajectory(group,method)
            result={'SKU':sku,'method':method,'source_rows':len(group),
                    'full_days':len(full),'scored_days':len(scored),**inventory_cost(scored)}
            results.append(result)
            for key,value in diag.items():diagnostics[method][key]+=value
    per=pd.DataFrame(results)
    summary=per.groupby('method',sort=False)[['holding_cost','stockout_cost','total_cost']].mean().reset_index()
    summary['n_sku_values']=df.SKU.nunique()
    summary['scope']='disclosed_subset_author_scoring'
    save_csv(per,out/'tables/offline_by_sku.csv')
    save_csv(summary,out/'tables/offline_summary.csv')
    dump_json(out/'logs/offline_diagnostics.json',{'source_rows':len(df),'unique_sku_values':int(df.SKU.nunique()),
        'row_order':'original within each SKU; no timestamps available', 'methods':diagnostics,
        'normal_formula7_scope':'sensitivity: only Normal predictions changed; author trajectory retained',
        'caveat':'Stored OPT column is evaluated under author checkpoint mechanics; not a certified lower bound here.'})
    from .plots import offline_chart
    offline_chart(summary,out/'figures/offline_costs.png')
    return summary
