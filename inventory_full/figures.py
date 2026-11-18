import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def save(fig,out,name):
    path=out/'figures';path.mkdir(parents=True,exist_ok=True)
    fig.savefig(path/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def table(ax,df,title):
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(0,1.03,title,fontsize=12,fontweight='bold')
    xs=[.0,.49,.74,.99]
    for x,h in zip(xs,['Method','Total cost','Holding cost','Stockout cost']):ax.text(x,.93,h,ha='left' if x==0 else 'right',fontsize=11)
    ax.plot([0,1],[.90,.90],color='black',lw=.7)
    base=float(df.loc[df.method=='OPT','total_cost'].iloc[0])
    for k,r in enumerate(df.itertuples(index=False)):
        y=.81-k*.10;w='bold' if r.method=='E2E_RNN' else 'normal'
        ax.text(0,y,r.method,fontweight=w,fontsize=11)
        total=f'{r.total_cost:,.2f}'+(f' (+{100*(r.total_cost/base-1):.1f}%)' if r.method!='OPT' else '')
        for x,s in zip(xs[1:],[total,f'{r.holding_cost:,.2f}',f'{r.stockout_cost:,.2f}']):ax.text(x,y,s,ha='right',fontsize=11,fontweight=w)
    ax.plot([0,1],[.15,.15],color='black',lw=.7)

def main_figures(out,cfg):
    df=pd.read_csv(out/'tables/costs_main.csv')
    order=['OPT','E2E_RNN','BM1','BM2','Normal','Gamma','E2E_GBM']
    df=df.set_index('method').loc[order].reset_index()
    f=pd.read_csv(out/'tables/figure2_forecasts.csv')
    fig=plt.figure(figsize=(17,8));gs=fig.add_gridspec(3,2,width_ratios=[1,1.5],wspace=.18,hspace=.4,left=.06,right=.98,bottom=.15,top=.85)
    fig.suptitle('Figure 2 reconstruction on SYNTHETIC continuous inventory panel',fontsize=18,y=.98)
    fig.text(.06,.92,f'{cfg["n_series"]:,} series | {cfg["test_days"]} test days | genuine policy-specific states and replenishment queues',fontsize=11,color='#8b3428')
    for k,i in enumerate([0,1,2]):
        a=f[f.series_id==i];future=a[a.relative_day>=0]
        ax=fig.add_subplot(gs[k,0]);ax.plot(a.relative_day,a.demand,color='#2375b5',lw=1.3,label='True demand')
        ax.plot(future.relative_day,future.q50,color='#ef8a36',lw=1.4,label='Predicted median')
        ax.fill_between(future.relative_day,future.q10,future.q90,color='#2375b5',alpha=.23,label='q10-q90')
        ax.axvline(0,color='#555555',ls='--',lw=.6);ax.set_ylim(bottom=0);ax.set_ylabel('Demand');ax.set_title(f'Synthetic series {i}',loc='left',fontsize=10)
        if k==0:ax.legend(fontsize=8)
        if k==2:ax.set_xlabel('Day relative to first test order')
    table(fig.add_subplot(gs[:,1]),df,'Costs over the same synthetic test panel')
    fig.text(.06,.065,'LSTM + global/local quantile decoder; forecast bands are model quantiles, not calibrated confidence intervals.',fontsize=10)
    fig.text(.06,.025,'New simulated results, NOT the original JD numbers. First three series shown without selection by accuracy.',fontsize=10,color='#8b3428')
    save(fig,out,'figure2')
    fig,ax=plt.subplots(figsize=(11,4));ax.bar(df.method,df.holding_cost,label='Holding');ax.bar(df.method,df.stockout_cost,bottom=df.holding_cost,label='Stockout')
    ax.set(ylabel='Mean total cost per series',title='Synthetic 30-day rolling inventory evaluation');ax.legend();ax.tick_params(axis='x',rotation=20)
    save(fig,out,'offline_costs')
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['main','forecast']:
        l=pd.read_csv(out/f'tables/train_{name}.csv');ax.plot(l.epoch,l.validation_loss,label=name+' validation')
    ax.set(title='Training diagnostics (synthetic)',xlabel='Epoch',ylabel='Joint loss');ax.legend();save(fig,out,'training_curves')
    fig,ax=plt.subplots(figsize=(13,7));ax.set_xlim(0,13);ax.set_ylim(0,7);ax.axis('off')
    boxes={'history':(0,5,2.8,1),'LSTM encoder':(3.5,5,2.8,1),'global/local decoder':(7,5,3.2,1),
           'demand heads':(10.5,5,2.2,1),'vendor history':(0,2.8,2.8,1),'VLT 50 / 20':(3.5,2.8,2.8,1),
           'fusion layers':(7,2.8,3.2,1),'order output':(10.5,2.8,2.2,1),'review R':(4,1,2,1),'net stock / position':(7.3,1,2.7,1)}
    for label,(x,y,w,h) in boxes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor='#e1f1f1',edgecolor='#1c7e82'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
    for start,end in [((2.8,5.5),(3.4,5.5)),((6.3,5.5),(6.9,5.5)),((10.2,5.5),(10.4,5.5)),((2.8,3.3),(3.4,3.3)),((6.3,3.3),(6.9,3.3)),((8.5,5),(8.5,3.9)),((10.2,3.3),(10.4,3.3)),((5,2),(7.5,2.7)),((8.5,2),(8.5,2.7))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.2})
    ax.text(.1,.35,'Figure 1 implementation map: synthetic 132D snapshot + known future plans. Shared decoder at encoder positions 20/24/28.',fontsize=10)
    save(fig,out,'figure1_architecture')

def matching(scores,out):
    t,c,m=scores;fig,axes=plt.subplots(1,2,figsize=(11,4))
    bins=np.linspace(min(t.min(),c.min()),max(t.max(),c.max()),25)
    for ax,x,title in zip(axes,[c,m],['Before matching','After propensity-score matching']):
        ax.hist(x,bins=bins,density=True,alpha=.5,label='Control');ax.hist(t,bins=bins,density=True,alpha=.5,label='Treatment')
        ax.set(title=title,xlabel='Estimated propensity score',ylabel='Density');ax.legend()
    fig.suptitle('Figure 3: SYNTHETIC intervention, pre-treatment matching');save(fig,out,'figure3_matching')

def appendix(tables,out,weights):
    df=tables['fraction'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(df.value,df.total_cost,'o-');axes[0].set(xlabel='Fraction of training series',ylabel='Test total cost')
    axes[1].plot(df.value,df.train_seconds,'o-');axes[1].set(xlabel='Fraction of training series',ylabel='Measured training seconds')
    fig.suptitle('Figure B.1: actual retraining on synthetic subsets');save(fig,out,'figureB1_data_size')
    df=tables['ratio'];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['total_cost','holding_cost','stockout_cost']):
        for method,g in df.groupby('method'):ax.plot(g.value,g[key],'o-',label=method)
        ax.set(xlabel='b / h',ylabel=key);ax.legend(fontsize=8)
    fig.suptitle('Figure B.2: relabel + retrain E2E for each cost ratio (synthetic)');save(fig,out,'figureB2_cost_ratio')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(seed,w) in zip(axes,weights):
        im=ax.imshow(w,aspect='auto',cmap='coolwarm');ax.set(title=f'Initialization seed {seed}',xlabel='Input feature',ylabel='VLT hidden neuron');fig.colorbar(im,ax=ax)
    fig.suptitle('Figure B.3: actually trained VLT first-layer weights');save(fig,out,'figureB3_initialization')
    for kind,title in [('layers','Fusion layer count'),('hidden','LSTM hidden size'),('lr','Learning rate')]:
        df=tables[kind];fig,ax=plt.subplots(figsize=(6,4));ax.plot(df.value,df.total_cost,'o-');ax.set(xlabel=title,ylabel='Synthetic test cost')
        if kind=='lr':ax.set_xscale('log')
        save(fig,out,'sensitivity_'+kind)
