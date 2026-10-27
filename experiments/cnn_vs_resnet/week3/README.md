# 第三周：CNN、ResNet 与 BN / 残差对照

学习指南：[第三周逐日任务](../../../docs/week3.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 `petvision`；首次 PyTorch 安装与排错见[第二周 Day 1](../../../docs/week2.md#environment)，本周检查和操作按[第三周指南](../../../docs/week3.md)完成。每次运行前：

```bash
conda activate petvision
python -m pip check
```

只在依赖发生变化时更新环境定义和快照。记录实际 GPU、显存、batch 与单 epoch 耗时；原始数据放在 `datasets/raw/`。本周实验目录已提供，配置和脚本按指南实现。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/inspect_cifar10.py` | 建立和检查固定划分 |
| Day 2–4 | `python -m pytest tests/test_resnet.py -q` | CNN/残差块/参考结构检查 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --smoke` | CNN 短运行 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --overfit-samples 64` | CNN 过拟合检查 |
| Day 2–5 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml` | BN on |
| Day 5 | `python scripts/train_cv.py --config configs/cv/cnn_no_bn_cifar10.yaml` | BN off |
| Day 4–5 | `python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml` | 残差 on |
| Day 5 | `python scripts/train_cv.py --config configs/cv/resnet_no_residual_cifar10.yaml` | 残差 off |
| Day 6 | `python scripts/evaluate.py --config configs/cv/cnn_cifar10.yaml --checkpoint outputs/cnn_bn_on/best.pt --split val` | BN on 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/cnn_no_bn_cifar10.yaml --checkpoint outputs/cnn_bn_off/best.pt --split val` | BN off 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/resnet_cifar10.yaml --checkpoint outputs/resnet_on/best.pt --split val` | 残差 on 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/resnet_no_residual_cifar10.yaml --checkpoint outputs/resnet_off/best.pt --split val` | 残差 off 独立验证 |

## 配置与结果

BN 是批归一化，residual 是残差连接；on/off 表示启用 / 关闭。variant 是实验变体名称，路径中的 `<variant>` 要替换成 cnn_bn_on、cnn_bn_off、resnet_on 或 resnet_off，不保留尖括号。卷积、池化、stage 和初始状态对齐的含义见[本周术语与约定](../../../docs/week3.md#cnn-terms)。

起始配置：seed=42、batch=64、20 epochs、AdamW lr=1e-3、weight_decay=1e-4、FP32；两对实验内部预算一致。训练/验证为 45k/5k，增强只作用训练。各组 experiment_dir 为 `experiments/cnn_vs_resnet/week3/runs/<variant>`，四组 checkpoint 输出目录分别为 `outputs/cnn_bn_on`、`outputs/cnn_bn_off`、`outputs/resnet_on`、`outputs/resnet_off`，best 按验证 accuracy 选择，持平保留较早一次；所有配置保存实际路径、参数量与初始化对齐方法。测试留到第五周。

CIFAR-10 加载函数、两个 transform 与检查脚本的接线见[Day 1 数据步骤](../../../docs/week3.md#cifar-data)，沿用第二周的 JSON / DataLoader 方法，分别处理 45k/5k/10k 三份数据。

配置中的 validation_size 和 num_workers 传给 build_cifar10_loaders，归一化 / 增强参数用于创建 transform；正式划分固定为 45000/5000。逐轮追加 history、耗时与峰值显存保存到 run_summary.json，具体见[资源记录步骤](../../../docs/week3.md#training-resources)。

初始状态按[训练前保存步骤](../../../docs/week3.md#initial-state)产生：Day 2 写 `outputs/cnn_bn_on/initial.pt`，Day 4 写 `outputs/resnet_on/initial.pt`，均在正式训练的第一次前向前保存，step=0。smoke/overfit 的模型不用于这些文件。

Day 5 在 BN off 配置设置 `training.init_checkpoint: outputs/cnn_bn_on/initial.pt`，在 residual off 配置设置 `training.init_checkpoint: outputs/resnet_on/initial.pt`。两组分别复制并核对共有层，保存各自 initial.pt 后，从新优化器、零进度开始训练；best/last 用于训练结果和评估。文件不存在时先补齐该对基线，不能替换成训练后权重。运行命令不变，这个字段由配置读取。数据顺序使用独立 Generator；相同 seed 只能作为随机设置记录，若未独立控制随机增强，不声称逐样本增强完全一致。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `cifar10_split.json` | Day 1：训练/验证固定索引 |
| `configs/` | Day 5：四组实际配置快照 |
| `results.csv` | Day 6：验证结果与预算 |
| `figures/bn_comparison.png` | BN 对照曲线 |
| `figures/residual_comparison.png` | 残差对照曲线 |
| `runs/<variant>/` | 各组配置快照、逐轮追加的 history、run_summary.json 与单组图 |
| 各组 `figures/mistakes_val.png` | Day 6：验证误分类，位于各自 experiment_dir |
| `outputs/{cnn_bn_on,cnn_bn_off,resnet_on,resnet_off}/` | 项目根下各组 initial（step=0）和 best/last checkpoint |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

