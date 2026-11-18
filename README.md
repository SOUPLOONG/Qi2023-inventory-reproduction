# Qi2023库存管理复现工程：现场训练与评估修正版 v4

本版修正V3中现场训练/测试条件不一致的问题，保留作者披露数据、离线论文模型、Figure 2及附录实验，并新增现场校准和四环境对照。**现场校准属于工程扩展，不能称为作者原算法的精确复现。** 模型是否降本以实际测试表为准，设置匹配不保证优于Normal。

## 从这里开始

1. 打开`results/standard/复现报告.html`查看全部图表；现场变化见`field_v4_comparison.png`和`field_v4_factorial.csv`。
2. 阅读`docs/现场代码修正_v4.md`了解修正理由、信息时序与方法边界。
3. 安装依赖后，在本文件所在目录运行：

```powershell
python -m pip install -r requirements.txt
python run_field_v4.py --profile quick
python run_field_v4.py --profile standard
python -m unittest discover -s tests -v
python verify_outputs.py --profile standard
python verify_field_v4.py --profile standard
```

已有模型会核验数据/源码/设置签名后重用。新环境完整运行用`python run_full.py --profile standard --stage all`；继承的离线模型及附录权重不必重新训练。若要全部重训可使用新输出目录和`--force`。更改现场训练预算使用`--field-epochs 30`，同时指定新输出目录以免混用缓存。

标准版离线预算10轮，新增现场两个校准模型各20轮；quick为2轮/4轮。CUDA可用则自动使用GPU，否则CPU。所有依赖见requirements，须先执行`python -m pip install -r requirements.txt`（其中lightgbm==4.6.0需另行安装，Windows下可从官网获取对应wheel）。

## 主要修改文件

|文件|修改|
|---|---|
|`inventory_full/engine.py`|提前期倍数与缺货模式独立配置；环境/特征一致性检查；拒绝不适用的闭式OPT；可插入校准策略|
|`inventory_full/field_calibration.py`|冻结MQ-RNN编码器；Normal残差校准头；可求导库存队列；现场实际成本训练；共同验证环境选轮次|
|`inventory_full/field_holdout.py`|三个固定独立面板仅评估，前后核验权重不变|
|`inventory_full/field.py`|接入校准策略，保留PSM、聚类回归和DID；保存逐日现场轨迹与初始状态|
|`run_full.py`、`run_field_v4.py`|完整入口和现场单独入口，现场预算独立设置|
|`tests/test_field_v4.py`|四环境模拟器一致性、有限差分、未来扰动、初始基线与环境保护测试|
|`verify_field_v4.py`|逐日/逐商品成本、丢失需求库存平衡、配对比较、缓存与独立评估核验|

## 数据和模型

- `data/author`：用户提供ZIP的9份原始CSV，保留原作者权利。
- `data/standard`：2,000条序列×420天=840,000行原合成面板及训练样本，沿用V3，未为获得更好排序重新生成。
- `data/independent_1000_20261020`等：三个固定种子的独立合成面板，每个1,000条序列，仅用于额外评估，不参与训练/选模型。
- `results/standard/models`：继承20个原网络检查点，新增`field_offline_control.pt`、`field_matched.pt`，以及真实LightGBM模型。
- 新现场模块的上下文维度149，另加3维可观测库存状态。原文132维输入的离线实现不变；新增上下文不能冒充原文私有132维定义。

校准训练成本期220—279，冻结编码器信息截至329，现场验证330—359，测试370—399；这是359日截止后的回顾建模。所有动作不使用真实未来需求/到货日。需求真实可观测、供应对订单数量外生、Normal替代私有京东算法等假设仍保留。

## 如何理解结果

`field_v4_factorial.csv`是同一批商品的配对比较，四个环境、四个策略均展示，不按效果筛选。两个校准头只有训练环境不同，结构/预算/目标现场验证环境相同；epoch0为Normal策略，也可被验证选择。

`table2_synthetic_field.csv`是匹配后的处理/对照样本比较，`table3`是含截距聚类回归，`table4`是DID。匹配组均值与全体同商品配对效果是不同统计量，不应混算。独立面板结果见`independent_holdout_summary.csv`。

V3测试期已用于问题诊断，所以新结果属于修正后的探索评估，不能包装为完全盲测。额外固定面板用于检查外部泛化；它们仍不代表真实京东现场。若新模型仍高于Normal，报告会保留该差异。

原论文精确数值仍缺完整JD数据、私有特征、权重、训练细节和现场算法。原V3说明保存在`docs/README_v3_历史说明.md`供追溯；当前运行方式和现场状态以本文件、V4修正说明及实际日志为准。
