# Qi et al. (2023) 库存管理论文：全文实验流程重建 v3

**先看范围：本工程完整执行合成数据上的方法重建，并独立复算作者披露的子样本。不是京东完整原始数据的精确数值复现。** 作者ZIP没有完整训练面板、网络源码及权重；生成更大数据可以补齐流程，不能补回真实业务分布或原始现场实验。

论文：Qi et al., *A Practical End-to-End Inventory Management Model with Deep Learning*, Management Science 69(2), 759–773, DOI [10.1287/mnsc.2022.4564](https://doi.org/10.1287/mnsc.2022.4564)。MQ-RNN结构参考 [Wen et al.](https://arxiv.org/abs/1711.11053)；实现为本项目独立编写。

## 1. 在哪里运行

把**整个文件夹**解压到本地，使用 Python 3.12 打开终端，并进入该文件夹。也可在 VS Code 中“打开文件夹”。无需把数据逐个导入，程序使用相对路径自动寻找 `data`。

推荐先看已生成的 `results/standard/复现报告.html`，再打开 `教学与运行.ipynb`。双击 Windows 的 `运行快速检查.bat` / `运行标准全部实验.bat` 也可执行；批处理使用UTF-8并保留中文路径兼容。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_full.py --profile quick --stage all
.\.venv\Scripts\python.exe run_full.py --profile standard --stage all
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

已安装依赖的电脑可直接 `python run_full.py --profile standard --stage all`。标准版自动选择可用CUDA，否则CPU。Windows包附带LightGBM 4.6.0；其他依赖仍须安装。若安装了CPU版PyTorch而需要GPU，请按 [PyTorch官方安装页](https://pytorch.org/get-started/locally/) 选择适配自身CUDA环境的命令。CPU能运行，耗时会增加。

## 2. 三种规模与分步运行

|配置|序列数×天数|日需求行数|每个网络轮数|用途|
|---|---:|---:|---:|---|
|quick|80×420|33,600|2|验证环境、快速演示；不评判论文结论|
|standard|2,000×420|840,000|10|本次实际完整运行配置|
|large|10,000×420|4,200,000|30|扩大数据和训练预算；本次未运行|

```powershell
python run_full.py --profile standard --stage generate
python run_full.py --profile standard --stage main
python run_full.py --profile standard --stage sensitivity
python run_full.py --profile standard --stage field
python run_full.py --profile standard --stage author
python run_full.py --profile standard --stage report
python run_full.py --profile large --stage all
```

`main`：E2E网络、相同结构预测网络、真实LightGBM、七策略滚动测试及Figure 1/2。
`sensitivity`：先确认主要模型，再运行Tables B.2—B.6和Figures B.1—B.3。
`field`：按12/88成本重训，模拟连续干预、PSM、t检验、含截距回归、DID和Figure 3。
`author`：仅重算作者披露CSV原口径。

修改预算示例：`--epochs 30 --output results/standard_30epochs --force`。可用 `--optimizer sgd --batch-size 64` 启用SGD动量0.85/论文批量；其余未公开训练细节仍属重建选择。不要将覆盖后的旧结果与不同配置混合。模型缓存同时核验数据、关键源码和训练设置，设置变化会拒绝加载，须指定新输出目录或明确 `--force`。

## 3. 实际交付物

- `data/author`：作者披露CSV原文件，未改造为“完整JD数据”。
- `data/standard`：完整生成面板、压缩CSV、实际提前期、训练样本与索引、来源和切分审计。
- `inventory_full`：新模型与逐日库存模拟器；`author_reference`：保留的作者口径复算实现。
- `results/standard/models`：真实训练的各模型权重、LightGBM文本模型、两个种子的VLT权重。
- `results/standard/tables`：主要成本、附录实验、逐商品成本、每轮训练损失、现场统计和匹配明细。
- `results/standard/trajectories`、`decisions`：各主要策略逐日库存/成本、逐单订货/到货记录。
- `results/standard/figures`：Figure 1/2/3、B.1/B.2/B.3及辅助诊断，PNG和SVG两种格式。
- `results/standard/复现报告.html`：可离线阅读，图表来自本次运行，不依赖互联网。
- `docs`：公式、数据字典、范围差异、作者口径和从零理解路线。
- `tests`：独立LP最优性核验、信息时序、fork因果性、模型保存、连续库存队列等验证。

## 4. 必须连同结果说明的差异

1. 原论文离线样本为24,333个食品类SKU；本工程是2,000个合成商品—仓库序列，并非这些SKU的替代真实记录。
2. 原论文训练/验证用前30天订单的80%/20%；本工程训练决策日180—260（标签截至279），验证决策日280—310（标签截至329），测试330—359。为避免标签跨期泄漏，采用了额外purge；数据生成、时间切分和类别均不相同。
3. 论文132维私有特征不能恢复，本工程公开重新定义132维快照及已知未来计划；模型宽度尽量对应，MQ-RNN解码细节、损失及权重、预测跨度、训练轮数等独立明确选择。默认Adam/批量256属于计算配置差异。
4. 合成现场只训练一个跨类别模型；论文现场按茶具和其他三级类别分别训练，京东私有现行算法也不可恢复，本工程对照使用Normal。现场2—8天交叉到货与混合缺货属于离线标签模型的压力测试，假定真实需求完全可观测；模拟DID不证明真实因果效应。
5. 对比基准OPT是已知未来的事后下界。有限期末处理与平稳策略不同；不能把全部差距解释为预测优劣。
6. 附录B.1列出的其他可选超参数范围是论文信息；本工程实际完整重训的是正文和Tables B.2—B.6报告的设置，不声称穷举Table B.1所有组合。两种种子的接近程度只描述这两次运行，不证明所有局部最优相同。

运行日志保留版本、设备、随机种子、数据与源码签名、训练样本和用时。没有为了贴近原文数值而填表、伪造效应或挑选测试分位数。跨硬件/版本浮点数可能略有变化，最终以你的运行日志为准。
