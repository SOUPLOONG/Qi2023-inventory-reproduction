"""Build an honest, portable Chinese audit report from actual run artifacts."""
from pathlib import Path
import html,json
import pandas as pd
from .core import dump,csv

def report(root,out,cfg,session):
    sections=[];md=[];v4=(out/'logs/field_v4_protocol.json').exists()
    def paragraph(title,text):
        sections.append(f'<h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>')
        md.extend(['## '+title,'',text,''])
    def table(name,title):
        p=out/'tables'/name
        if p.exists():
            df=pd.read_csv(p);sections.append(f'<h3>{html.escape(title)}</h3>'+df.to_html(index=False,float_format=lambda x:f'{x:.4f}'))
            md.extend(['### '+title,'',f'[CSV](tables/{name})','',df.to_string(index=False),''])
    def figure(stem,title):
        p=out/'figures'/f'{stem}.png'
        if p.exists():
            sections.append(f'<h3>{html.escape(title)}</h3><a href="figures/{stem}.svg"><img src="figures/{stem}.png" alt="{html.escape(title)}"></a>')
            md.extend([f'### {title}','',f'![{title}](figures/{stem}.png)',''])
    paragraph('本次结果的含义','本工程覆盖论文主要公式、预测与决策模型、多期离线评估、参数和数据量敏感性分析，以及模拟干预的匹配、回归、双重差分流程。完整训练实验使用自生成数据；作者披露数据另行计算。它是可运行的方法重建，不能等同于原始京东数据上的逐数值精确复现，也不能替代论文原始实地实验。')
    if v4:
        paragraph('V4现场修正的性质','离线模型及附录实验保留V3结果，新增两个现场校准模型。匹配模型在2—8天提前期、可能交叉、混合积欠/丢失销售环境中，以实际滚动成本训练；原闭式标签未用于这个环境。校准头冻结离线MQ-RNN表示并修正Normal基库存目标，这是工程扩展，不是原作者现场神经网络的精确恢复。')
        fpath=out/'tables/field_v4_factorial.csv'
        if fpath.exists():
            f=pd.read_csv(fpath);g=f[f.environment=='lead2_hybrid'].set_index('policy')
            old=float(g.loc['V3_Transfer','total_cost']);new=float(g.loc['V4_MatchedCalibrator','total_cost']);normal=float(g.loc['Normal','total_cost'])
            paragraph('V4同商品测试：改进与仍未解决的差距',f'全部商品配对比较：Normal成本{normal:.2f}，旧迁移策略{old:.2f}，新匹配校准策略{new:.2f}。新版相对旧迁移策略变化{100*(new/old-1):+.2f}%，相对Normal变化{100*(new/normal-1):+.2f}%。因此即使训练条件已修正，仍不能声称恢复了论文现场降本结论；其他控制策略及独立面板也全部公开。')
    paragraph('实际数据与运行环境',f"配置 {session['profile']}；{cfg['n_series']:,} 个合成商品—仓库序列 × {cfg['days']} 天 = {session['daily_rows']:,} 行日需求；训练样本 {session['training_samples']:,}，验证样本 {session['validation_samples']:,}。随机种子 {cfg['seed']}。设备 {session['device']}，{session.get('gpu') or 'CPU'}。本次入口用时 {session['elapsed_seconds']:.1f} 秒（重用缓存时不代表原训练总时长，逐模型时间见 tables/train 与 logs/model）。每个非默认设置均实际重新训练，最优轮次由验证集选择。")
    if (out/'tables/costs_main.csv').exists():
        cost=pd.read_csv(out/'tables/costs_main.csv').set_index('method')
        e=float(cost.loc['E2E_RNN','total_cost']);normal=float(cost.loc['Normal','total_cost'])
        text=f'离线E2E平均30天总成本{e:.2f}，Normal为{normal:.2f}，相对变化{100*(e/normal-1):+.2f}%。'
        fp=out/'tables/table2_synthetic_field.csv'
        if fp.exists():
            f=pd.read_csv(fp).set_index('metric').loc['total_cost'];change=float(f.relative_change_pct)
            text+=f'模拟现场处理组总成本{float(f.treatment_mean):.2f}，对照组{float(f.control_mean):.2f}，相对变化{change:+.2f}%。'
            if change>=0:text+='本次模拟现场没有重现论文的总成本下降结论；程序执行通过不代表论文所有实证结论已得到重复支持。'
        paragraph('实际结论：包括未复现的结果',text)
    if v4:
        paragraph('如何读新版现场比较','下面是同一批商品、同一需求和初始库存/在途订单的配对比较，不能与后面的匹配组均值混算。V3迁移、离线条件校准、匹配条件校准和Normal全部保留。旧测试期已用于问题诊断，属于探索评估；额外三组固定独立面板仅用于评估，没有按结果选权重。训练条件匹配不保证测试成本更低。')
        table('field_v4_factorial.csv','V4：四种环境×四种策略的固定配对实验')
        table('independent_holdout_summary.csv','V4：三个固定独立面板的额外评估')
    paragraph('先看哪些结果','Figure 2 展示固定前三个商品的真实需求、神经网络预测中位数、10%—90%预测分位带，以及七种方法的实际模拟库存成本。阴影是需求预测分位带，不是均值置信区间。成本表由全部测试商品汇总，每种策略独立滚动自己的库存和在途订单；不按效果挑选商品，不把原论文数值填进模拟结果。')
    # Names are discovered from exported files to keep the report aligned with actual artifacts.
    for p in sorted((out/'figures').glob('*.png')):
        if 'figure1' in p.stem.lower() or 'figure2' in p.stem.lower():figure(p.stem,p.stem)
    table('costs_main.csv','离线七种方法：真实运行成本，SYNTHETIC')
    taxonomy=[]
    groups={'Tea set':['Tea set combination','Tea cup','Tea kettle','Tea tray','Tea can','Tea bowl','Tea accessories','Tea-ware decoration','Coffee set','Tea travel set'],
        'Pastry essentials & seasoning':['Baking supplies','Flour','Mixed grains','Rice','Oil','Seasonings','Dry foods','Convenience foods']}
    for group,items in groups.items():
        taxonomy.extend({'second_level_category':group,'third_level_category':item,
            'source':'Paper Table 1, p.767, reference transcription only'} for item in items)
    csv(pd.DataFrame(taxonomy),out/'tables/table1_paper_categories_reference.csv')
    table('table1_paper_categories_reference.csv','Table 1：原文类别明细转录（描述信息，不是模型运行结果）')
    table('table1_synthetic_composition.csv','附加统计：本工程合成类别计数，不等同于原文Table 1')
    hyper=[('MQRNN hidden size','50','30,40,50,60','50'),('VLT layers','50,20','100/20;50/20;30/20','50,20'),
        ('Integration layers','100,100','1/2/3/4 layers of100','100,100'),('Learning rate','0.001','0.0001,0.001,0.01','0.001'),
        ('Learning-rate decay','1e-4','1e-5,1e-4,1e-3','0; no decay in this run'),
        ('Momentum','0.85','0.8,0.85,0.9','0.85 if SGD; not applicable to Adam'),
        ('Mini-batch size','64','64,128,256',str(cfg['batch_size'])),('Initialization','Gaussian(0,0.01)','Gaussian,Uniform','Gaussian(0,0.01); output biases calibrated'),
        ('Activation','ReLU','ReLU,tanh','ReLU; softplus positive output; standard LSTM gates'),('Dropout','0.2','0.1,0.2,0.3,0.4','0.2'),
        ('Optimizer','SGD family discussed','details incomplete',cfg['optimizer']),('Epochs','not reported','not reported',str(cfg['epochs'])),
        ('Forecast horizon','not reported','not reported',str(cfg['horizon'])),('Quantiles','count ambiguity in text','see paper sec4.2','0.1,0.5,0.6,0.7,0.8,0.9,0.95 + mean'),
        ('Loss and lambda weights','generic losses; small positive lambdas','not specified','SmoothL1 + 0.1 demand loss + 0.1 VLT MSE')]
    csv(pd.DataFrame(hyper,columns=['parameter','paper_reference','paper_reported_range','implementation']),out/'tables/table_B1_hyperparameters.csv')
    table('table_B1_hyperparameters.csv','Table B.1 对应：论文与重建超参数对照')
    paragraph('模型与公式对应','式(1)—(3)：以逐日持有成本和缺货成本构成多期目标，并维护净库存与订单到货队列；本文实现学习策略而非声称精确求解所有随机动态规划。式(4)—(6)：在离线无订单交叉条件下，用实际未来需求构造条件事后最优订货标签，训练时使用已成熟标签；独立线性规划测试验证其有限时域OPT。式(7)：Normal 使用随机提前期方差项；式(8)：Gamma 使用形状—尺度参数近似。式(9)—(10)：BM1累计需求分位数订货与BM2逐日预测路径的多期成本最优化。式(11)：含截距及干预前协变量的处理组回归；后续DID按原商品ID聚类。现场混合积欠/丢失销售是依据正文描述单独定义的模拟规则。原文推导和识别假设的解释见项目根目录 docs/公式与代码对应.md。')
    paragraph('论文未公开细节与本工程的明确选择','作者包缺少完整训练面板、原神经网络训练源码、权重及全部超参数。此处独立实现LSTM的MQ-RNN、全局/共享局部解码器、三个训练fork、VLT 50→20网络、100→100融合网络及真实LightGBM。累计需求分位数由单独监督头学习，没有把每日分位数直接相加。132维输入的组成是本工程公开定义，不声称与私有特征完全相同。预测跨度20天、SmoothL1订货损失加预测/VLT辅助损失及其权重、7个预测分位点加均值、Adam、批量256均是公开记录的重建选择；默认训练轮数见配置。论文未报告的选择不作为原作者设定。')
    paragraph('时间顺序、信息集与期末效应','定标只用第0—119天；训练标签最晚第279天，验证标签最晚第329天，测试第330—359天。在线输入只含已发生需求、已完成供应商记录、静态属性及事先已知促销/日历。当前与下一单真实提前期只进入训练标签或模拟器，不作为在线预测特征。每个策略执行到货→订货→需求→计费，不重置历史缺货，订货非负。OPT截断最后到货区间，BM截断保护期，平稳策略没有专门终端价值训练，因此与OPT差距包含期末效应，不能全部归因于模型预测质量。')
    for p in sorted((out/'figures').glob('*.png')):
        if 'figure1' not in p.stem.lower() and 'figure2' not in p.stem.lower():figure(p.stem,p.stem)
    for name,title in [('table_B2_layers.csv','Table B.2 层数'),('table_B3_neurons.csv','Table B.3 隐藏单元'),('table_B4_learning_rate.csv','Table B.4 学习率'),('table_B5_data_size.csv','Table B.5 数据量'),('table_B6_initialization.csv','Table B.6 初始化'),('table_cost_ratio.csv','Figure B.2 成本比对应数据'),('table2_synthetic_field.csv','Table 2 模拟干预处理—对照'),('table3_synthetic_regression.csv','Table 3 含截距控制变量回归'),('table4_synthetic_did.csv','Table 4 两期DID'),('field_covariate_balance.csv','匹配前后协变量平衡')]:table(name,title)
    field_text='这是模拟干预而非京东现场。第300—329天预热、330—359天干预前、360—369天过渡，连续传递库存和在途订单，370天切换策略并观察至399天。PSM只用干预前变量，有放回匹配按原商品ID聚类。'
    if v4:
        field_text+='V4校准成本训练220—279，编码器信息截至329，两个校准头均按330—359同一现场验证环境选轮次。匹配头训练/验证/测试均采用2—8天与hybrid，h=12、b=88；另一个头仅训练环境不同，作为控制。策略只接收可观测库存与在途总量，实际未来到货日只用于模拟器。'
    else:field_text+='V3仅按88/12重训离线标签，2—8天交叉到货和混合丢失需求属于分布外压力测试。'
    field_text+='假定包括未满足需求在内的真实历史需求可观测。现场费用按每商品30天求和；Normal替代京东私有算法，共享模型替代原文分品类训练。两期DID不能检验平行趋势，合成显著性不证明真实业务因果效果。'
    paragraph('模拟现场实验的解释边界',field_text)
    author=out/'author_disclosed'
    if author.exists():
        paragraph('作者披露数据：独立保留的原口径','author_disclosed 子目录按作者公开notebook口径重算现有预测和现场样本，data/author保留原始CSV。该口径存在负补货、库存处理和Normal方差等问题，详见 docs/作者数据与口径.md，不能与修正后的新模拟成本混为一谈。数据量和截取窗口也不同于论文整体现有结果，因此仍不会逐项等于原Figure 2。')
        for p in sorted(author.rglob('*.csv')):
            rel=p.relative_to(out).as_posix();sections.append(f'<p><a href="{rel}">{html.escape(rel)}</a></p>')
    paragraph('如何重复与核验','在项目根目录执行 python run_full.py --profile standard --stage all。quick用于安装验证；large生成一万条序列。训练、验证、测试、敏感性、现场、图表和报告可按stage运行。数据与模型有签名，设置改变会拒绝误用缓存，使用不同目录或--force明确重跑。测试执行 python -m unittest discover -s tests -v。模型权重、每轮训练日志、分位点验证选择、逐商品成本、逐日库存、逐单到货、匹配样本与回归系数均保存，便于检查每一项结果。')
    expected=['costs_main.csv','table1_synthetic_composition.csv','table1_paper_categories_reference.csv','table_B1_hyperparameters.csv','table_B2_layers.csv','table_B3_neurons.csv','table_B4_learning_rate.csv','table_B5_data_size.csv','table_B6_initialization.csv','table_cost_ratio.csv','table2_synthetic_field.csv','table3_synthetic_regression.csv','table4_synthetic_did.csv']
    if v4:expected+=['field_v4_factorial.csv','independent_holdout_summary.csv']
    status={n:(out/'tables'/n).exists() for n in expected}
    dump(out/'logs/coverage_manifest.json',{'source':'synthetic method reconstruction','expected_tables':status,'all_expected_tables_present':all(status.values()),'figures':[p.name for p in sorted((out/'figures').glob('*.png'))],'not_exact_JD_replication':True})
    paragraph('完整性检查',f"本次目录应有的 {len(expected)} 张主表/附录表：已生成 {sum(status.values())} 张。完整性清单见 logs/coverage_manifest.json。该检查只证明产物存在，数学、时间顺序和结果来源还须结合测试与协议核验。")
    title='Qi 2023：'+('V4现场训练与评估修正报告' if v4 else '全文实验流程重建报告')+'（合成数据）'
    (out/'复现报告.md').write_text('# '+title+'\n\n'+'\n'.join(md),encoding='utf-8')
    (out/'复现报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1200px;margin:36px auto;padding:0 24px;color:#183047}h1,h2{color:#144d72}h2{margin-top:38px}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px}td,th{padding:7px 12px;border:1px solid #cdd8df;text-align:right}img{max-width:100%;height:auto}p{max-width:100ch}a{color:#08749d}</style><h1>'+title+'</h1>'+''.join(sections)+'</html>',encoding='utf-8')
