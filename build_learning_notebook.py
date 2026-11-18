"""Regenerate the small teaching notebook; does not execute full training."""
from pathlib import Path
import json

def md(s):return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)}
cells=[md('''# 从零理解与运行 Qi 2023 库存模型

这里每个小例子可逐格执行。完整实验已在 `results/standard`；大规模训练通过统一入口运行。V4现场校准是独立工程扩展，方法与限制见README及docs/现场代码修正_v4.md。所有生成数据明确是合成数据，不能用于声称重现京东原始数值。

先选择本项目的Python环境为Notebook内核。若从其他目录打开，先把下格的ROOT改为解压文件夹。'''),code('''from pathlib import Path
import sys, os
ROOT = Path.cwd()
if not (ROOT / 'run_full.py').exists():
    raise RuntimeError('请将Notebook工作目录设为本项目根目录')
sys.path.insert(0, str(ROOT))
if os.name == 'nt': sys.path.insert(0, str(ROOT / '.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work' / 'notebook_cache'))
import numpy as np
import pandas as pd
from IPython.display import display, Image
from inventory_full.core import label, advance
'''),md('''## 1. 先只做库存加减

`I`是净库存。`max(I,0)`取正库存，`max(-I,0)`取缺货量。持有1单位一天付h，缺1单位一天付b。积欠不会在第二天自动消失。'''),code('''stock = np.array([10.0])
rows=[]
for day, (arrival, demand) in enumerate([(0,3),(0,5),(17,4),(0,2),(0,7),(0,6)]):
    stock, H, B, lost = advance(stock, np.array([arrival]), np.array([demand]), np.zeros(1), h=1,b=9)
    rows.append([day,arrival,demand,float(stock[0]),float(H[0]),float(B[0])])
display(pd.DataFrame(rows,columns=['day','arrival','demand','end_inventory','holding_cost','backlog_cost']))
'''),md('''## 2. 闭式标签为什么是17

订单在第0天下达、第2天到货，下次到货第6天。当前订单控制第2—5天。α=9/(9+1)=0.9，floor(0.9×4)=3，因此覆盖至第5天的累计需求。27−10=17。

先用公式，再把0—35个单位逐一试一遍。枚举是这个小例子的独立检验；全局多订单检验在tests中使用线性规划。'''),code('''d = np.array([3,5,4,2,7,6],float)
a_star = label(d, stock=10, lead=2, next_arrival=6, b=9,h=1)
def interval_cost(a):
    inv=10.0; total=0.0
    for t,dt in enumerate(d):
        if t==2: inv+=a
        inv-=dt
        total+=max(inv,0)+9*max(-inv,0)
    return total
grid = np.arange(36)
print('公式标签:',a_star,'枚举最优:',grid[np.argmin([interval_cost(a) for a in grid])])
assert interval_cost(a_star) == min(interval_cost(a) for a in grid)
'''),md('''## 3. 从零看一次求导

设预测=参数w×输入x，损失=(预测−答案)²。对w求导得到2×(wx−答案)×x。导数为负说明w增加一点会让损失下降。PyTorch可自动求导，但它求的是你定义的损失，不能保证业务成本或测试效果一定改善。'''),code('''import torch
w=torch.tensor(1.0,requires_grad=True)
x=torch.tensor(2.0); answer=torch.tensor(6.0)
loss=(w*x-answer)**2
loss.backward()
print('自动梯度:',w.grad.item(),'手算:',2*(w.item()*x.item()-answer.item())*x.item())
with torch.no_grad(): w -= 0.1*w.grad
print('更新参数:',w.item(),'更新后损失:',((w*x-answer)**2).item())
'''),md('''## 4. 查看真实生成的数据和模型结果

训练输入与未来标签分别保存。模型只接收INPUTS，不能把`actual_future`、`Lnext`等当作在线输入。'''),code('''import json
profile='standard'
data=ROOT/'data'/profile
out=ROOT/'results'/profile
display(pd.read_csv(data/'daily_panel.csv.gz',nrows=12))
print(json.loads((data/'sample_audit.json').read_text('utf-8')))
display(pd.read_csv(out/'tables/costs_main.csv'))
display(Image(filename=str(out/'figures/figure2.png')))
'''),md('''## 5. 检查成本来自哪里

一行决策含订货量和到货日；每天都有需求、收货、净库存、持有及缺货成本。把每天的费用相加应与逐商品成本一致。'''),code('''orders=pd.read_csv(out/'decisions/E2E_RNN_orders.csv.gz')
daily=pd.read_csv(out/'trajectories/E2E_RNN_daily.csv.gz')
display(orders[orders.series_id==0])
display(daily[daily.series_id==0].head(12))
cost0=daily.loc[daily.series_id==0,['holding_cost','stockout_cost']].sum().sum()
per=pd.read_csv(out/'tables/costs_by_series_main.csv')
expected=per.loc[(per.series_id==0)&(per.method=='E2E_RNN'),'total_cost'].iloc[0]
np.testing.assert_allclose(cost0,expected)
print('商品0的总成本核验:',cost0)
'''),md('''## 6. 逐步重跑

下格默认不启动长训练。将RUN设为True即可运行快速版完整流程。使用`sys.executable`保证命令与Notebook是同一Python环境。完整标准版把quick改为standard；学习率、层数等实验通过sensitivity阶段自动重训。'''),code('''import subprocess
RUN=False
if RUN:
    subprocess.run([sys.executable,'run_full.py','--profile','quick','--stage','all'],cwd=ROOT,check=True)
'''),md('''## 7. 附录与模拟现场

注意：p值显著不意味着模拟结果能代表京东。现场对照是Normal替代算法，供应商交叉到货是分布外压力测试。DID只在本生成与处理分配机制内解释。'''),code('''display(pd.read_csv(out/'tables/table_B5_data_size.csv'))
display(pd.read_csv(out/'tables/table4_synthetic_did.csv'))
display(Image(filename=str(out/'figures/figure3_matching.png')))
''')]
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells):c['id']=f'qi2023-{i:02d}'
Path(__file__).with_name('教学与运行.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
