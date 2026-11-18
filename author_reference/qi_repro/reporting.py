import numpy as np
import pandas as pd
from .common import save_csv,dump_json
from .inventory import hindsight_label,simulate_backorders


def theory_examples(data_root,out):
    d=np.array([2,4,3,5.]);I=3;h=1;b=2
    label=hindsight_label(d,I,h,b)
    arrivals=np.array([label['order'],0,0,0.])
    inv,cost=simulate_backorders(d,I,arrivals,h,b)
    frame=pd.DataFrame({'day':np.arange(1,5),'demand':d,'cumulative_demand':np.cumsum(d),'arrivals':arrivals,
        'end_inventory':inv,'holding_cost':h*np.maximum(inv,0),'stockout_cost':b*np.maximum(-inv,0)})
    save_csv(frame,out/'tables/theory_example.csv')
    compare=[]
    for a in range(16):
        _,c=simulate_backorders(d,I,[a,0,0,0],h,b)
        compare.append({'order':a,**c})
    save_csv(pd.DataFrame(compare),out/'tables/theory_order_comparison.csv')
    dump_json(out/'logs/theory_example.json',{'label':label,'cost':cost,'initial_stock':I,'h':h,'b':b,
        'next_arrival_after_interval':True,'expected_order':6,'expected_cost':20})
    sales=pd.read_csv(data_root/'E2E-model-data/Data/rdc_sales_1320.csv')
    examples=[]
    for _,row in sales.iterrows():
        demands=row.iloc[1:5].to_numpy(dtype=float)
        lab=hindsight_label(demands,3,1,2)
        examples.append({'source_id':row.iloc[0],'source_columns':','.join(sales.columns[1:5]),
            'demand_values':','.join(map(str,demands)),'ASSUMED_initial_inventory':3,'ASSUMED_h':1,'ASSUMED_b':2,
            'ASSUMED_interval_days':4,'label_order':lab['order'],
            'scope':'real disclosed sales with assumed inventory/arrivals/costs; NOT reconstructed historical optimum'})
    save_csv(pd.DataFrame(examples),out/'tables/real_sales_assumed_label_examples.csv')
    from .plots import example_chart
    example_chart(d,inv,out/'figures/theory_inventory.png')


def table_md(frame,columns=None):
    if columns is not None:frame=frame[columns]
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def make_report(out,completed,session):
    lines=['# 本次运行结果与公式理解','',
        '本报告只引用本次请求的阶段；公开子样本分析和合成数据训练分开解释。',
        '实际完成：'+', '.join(completed)+f'；耗时{session["elapsed_seconds"]:.1f}秒。','']
    if 'theory' in completed:
        lines+=['## 式(1)、式(2)、定理1','',
            '例子需求2、4、3、5，初始净库存3，h=1、b=2。定理标签订6件，逐日库存7、3、0、−5，总成本20。',
            '先看tables/theory_example.csv，再看theory_order_comparison.csv，比较为什么补齐全部需求的成本反而更高。','']
    if 'offline' in completed:
        df=pd.read_csv(out/'tables/offline_summary.csv')
        lines+=['## 公开子样本的离线评价','',
            '使用3,000行输入，按2,607个不同SKU字段值分组，以原检查点计分口径求组均值。以下成本并非正式版全样本图2数字。',
            'OPT、E2E_RNN和GBM来自作者已披露输出，当前阶段没有重新训练这些模型。',
            table_md(df,['method','holding_cost','stockout_cost','total_cost']),'',
            'Normal_formula7仅替换正态基准的分位数和方差公式，轨迹仍按原包处理，属于单项敏感性分析。',
            '原包轨迹会产生负补货、清除旧负库存或跳过部分时段；详见logs/offline_diagnostics.json。不要把该列OPT当作所有其他方法的严格下界。','']
        import json
        diag=json.loads((out/'logs/offline_diagnostics.json').read_text('utf-8'))
        neg=diag['methods']['E2E_RNN']['negative_orders']
        lines += [f'本次E2E原包补货计算实际出现{neg}次负值；本表是忠实代码复算，不能直接解释为满足a≥0约束的可执行策略表现。','']
    if 'field' in completed:
        df=pd.read_csv(out/'tables/field_summary.csv')
        lines+=['## 现场公开子样本','',table_md(df,['specification','group','period','n','total_cost']),'',
            'author_checkpoint保留h=1、b=9和原检查点计费规则；另一规格只改为正文现场h=12、b=88并在回归中加截距。',
            '每张表只有500行，窗口长度不等；这里计算记录成本之和再取均值，不称为完整现场实验复现。',
            'field_ttests.csv、field_regressions.csv和field_did.csv分别对应组间检验、式(11)和原包前后比较。',
            'DiD中的legacy_rowwise_p忠实保留原包按行位置作差的t检验；尚未确认跨文件匿名ID的实体连接，不把该检验宣称为可靠的面板因果估计。',
            '匹配程序实际按原始特征距离选择候选对照，倾向得分用于展示；matching_indices.csv记录有放回匹配结果。','']
    if 'neural' in completed:
        df=pd.read_csv(out/'tables/neural_demo_summary.csv')
        info=__import__('json').loads((out/'logs/neural_demo.json').read_text('utf-8'))
        lines+=['## 合成数据上的模块化网络训练','',
            '**本节全部是合成数据上的方法重建，不是京东原始训练实验。**',
            f'固定种子{info["seed"]}，训练{info["train_n"]}例、验证{info["validation_n"]}例、测试{info["test_n"]}例；最佳验证轮次{info["best_epoch"]}。',
            table_md(df,['method','holding_cost','stockout_cost','total_cost']),'',
            '网络包含需求GRU、多步分位数输出、提前期模块和补货输出。输入仅为历史需求与促销、商品类别、历史提前期、下单时库存和审核间隔。',
            '真实未来需求、实际本单提前期和下一单到货日仅用于制作标签与评价。',
            '这里使用独立的单次决策场景；没有声称完成真实业务的多订单闭环训练。',
            '模型、损失和优化器中的补充选择已写入docs/implementation_choices.md。训练损失下降和测试表现均按实际结果报告，不保证E2E优于演示PTO。','']
    lines+=['## 如何边读边运行','',
        '1. 先打开00_从公式到代码.ipynb，执行库存记账和标签例子。',
        '2. 运行offline阶段，查看给定订货预测如何形成成本；与正文图2区分。',
        '3. 运行field阶段，理解现场指标和式(11)，核对成本参数。',
        '4. 运行neural阶段，观察式(4)、式(6)的标签、三个输出和训练过程。',
        '5. 所需真实公开数据已在data/author；演示数据生成器及已生成数据在data/synthetic。','',
        '完整训练数据、原MQRNN源码、确切主损失和λ仍未获得，因而本包不宣称复原正式论文全部表图和网络权重。']
    (out/'运行结果与理解说明.md').write_text('\n'.join(lines),encoding='utf-8')
