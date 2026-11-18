"""Standalone PNG artifacts; use English plot labels for portable fonts."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finish(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def offline_chart(summary,path):
    df=summary.set_index('method').reindex(['OPT','E2E_RNN','Bm1','Bm2','Normal','Gamma','gbm','Normal_formula7'])
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(df.index,df.holding_cost,label='Holding',color='#128288')
    ax.bar(df.index,df.stockout_cost,bottom=df.holding_cost,label='Stockout',color='#dc8251')
    ax.set(ylabel='Mean sum of scored costs per SKU value',title='Disclosed subset: original checkpoint scoring')
    ax.tick_params(axis='x',rotation=25);ax.legend();ax.grid(axis='y',alpha=.15)
    fig.text(.01,.005,'Stored predictions; Normal_formula7 changes only the Normal prediction formula.',fontsize=9)
    finish(fig,path)


def matching_chart(treated,candidates,matched,path):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    bins=np.linspace(min(treated.min(),candidates.min()),max(treated.max(),candidates.max()),30)
    for ax,controls,title in zip(axes,[candidates,matched],['Before matching','After raw-feature matching']):
        ax.hist(controls,bins=bins,density=True,alpha=.5,label='Control',color='#819aaa')
        ax.hist(treated,bins=bins,density=True,alpha=.5,label='Treatment',color='#128288')
        ax.set(title=title,xlabel='Fitted propensity score',ylabel='Density');ax.legend()
    finish(fig,path)


def learning_chart(log,path):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(log.epoch,log.train_loss,label='Train');ax.plot(log.epoch,log.validation_loss,label='Validation')
    ax.set(title='Synthetic training demonstration',xlabel='Epoch',ylabel='Joint normalized loss')
    ax.legend();ax.grid(alpha=.15);finish(fig,path)


def example_chart(demand,inventory,path):
    fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
    axes[0].bar(np.arange(len(demand)),demand,color='#819aaa');axes[0].set_ylabel('Demand')
    axes[1].plot(inventory,'o-',color='#128288');axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(xlabel='Day (zero based)',ylabel='End-of-day net stock')
    axes[0].set_title('Equation (1): full backorders; arrivals before demand')
    finish(fig,path)
