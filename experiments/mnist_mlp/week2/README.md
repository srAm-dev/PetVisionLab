# 第二周：PyTorch、MNIST MLP 与完整训练流程

学习指南：[第二周逐日任务](../../../docs/week2.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 Conda 环境 `petvision`。首次安装、驱动检查和 CUDA 排错按[第二周 Day 1](../../../docs/week2.md#environment)完成；本地包的 editable 安装按[Day 2](../../../docs/week2.md#day-2)完成。

```bash
conda activate petvision
python -m pip check
```

环境以 Python 3.12、NumPy、Matplotlib、PyYAML、pytest 和配套 torch / torchvision 为基础。

已有 [GPU 检查记录](environment.txt)：PyTorch `2.14.1+cu130`、torchvision `0.29.1+cu130`、PyTorch CUDA `13.0`、RTX 4060 Laptop GPU，总显存 `8.00 GiB`，当时可用 `6.93 GiB`，2×2 GPU 矩阵乘法已通过。

以上是历史记录，可用显存随运行状态变化；完整安装命令（含包索引地址）和驱动版本尚待补记，不能从版本号反推。根 `environment.yml` 是当前 Linux 完整快照，尚不能单独保证恢复相同 GPU wheel。依赖变化后保存本周记录：

```bash
conda list > experiments/mnist_mlp/week2/conda-packages.txt
conda list --explicit > experiments/mnist_mlp/week2/conda-explicit.txt
python -m pip freeze > experiments/mnist_mlp/week2/pip-freeze.txt
```

Conda 显式清单记录 Conda 包，pip 补充包还需依赖版本与实际安装命令。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。当前已有 Day 1 两个脚本、包配置、Day 2 的 MNIST 加载与检查，以及 Day 3 的固定结构 MLP 和三个基础测试；训练 / 评估入口仍需实现，表中后续参数是目标接口。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/check_environment.py --output-dir experiments/mnist_mlp/week2` | 打印并记录环境 |
| Day 1 | `python scripts/torch_basics.py` | Tensor 与 autograd 检查 |
| Day 2 | `python scripts/inspect_mnist.py` | 固定划分与数据检查 |
| Day 3–6 | `python -m pytest tests/test_cv_training.py -q` | 关键形状、梯度、保存加载 |
| Day 4 | `python scripts/train_cv.py --help` | 核对配置参数与互斥检查模式 |
| Day 4 | `python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --smoke` | 两批次 smoke |
| Day 4 | `python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --overfit-samples 64` | 固定小样本过拟合 |
| Day 5 | `python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml` | 接通保存后，从新模型正式训练 5 epochs |
| Day 5 | `python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split val` | 独立加载验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split test` | 冻结后测试 |

## 配置与结果

基线：seed=42，官方训练集 55,000/5,000，MLP 784→256→128→10，Adam lr=1e-3，batch=128，epochs=5，FP32。训练从新模型起步，按验证准确率选择 best；测试集只在配置冻结后使用。命令中的 output_dir、experiment_dir 和 split 文件路径在 YAML 中明确保存。

MNIST 下载、固定划分与 DataLoader 见[Day 2 数据步骤](../../../docs/week2.md#mnist-data)，训练对象接线见[Day 4](../../../docs/week2.md#training-objects)，实际保存 / 加载操作见[Day 5](../../../docs/week2.md#checkpoint-io)。

Day 2 数据条件：`data_dir=datasets/raw`，`split_path=experiments/mnist_mlp/week2/mnist_split.json`，seed=42，batch_size=128，num_workers=0，drop_last=False；预处理依次为 ToTensor 与 Normalize(mean=(0.5,), std=(0.5,))。训练打乱批次顺序，验证和测试按固定顺序取样；划分不随每轮训练改变。

Day 3 检查记录（2026-10-24）：运行 `python -m pytest tests/test_cv_training.py -q -s`，结果为 `3 passed`。CPU 前反向检查通过，设备测试在 RTX 4060 Laptop GPU（`cuda:0`）上通过。B=1/4 时 logits 分别为 `[1,10]` / `[4,10]`，损失为有限标量，所有可训练参数的梯度存在且有限；参数量为 235146。

手写三分类交叉熵检查：正确类 logit 从 0 提高到 2 时，loss 从约 1.098612 降至 0.239545。模型目前使用固定尺寸的 `MLP()`，Day 4 再扩展配置参数。

完整 YAML 见[Day 4 配置示例](../../../docs/week2.md#mnist-config)，参数解析见[最小入口示例](../../../docs/week2.md#cli-config)。统一使用 `data.root`、`data.split_path`、`data.batch_size` 和 `training.learning_rate`；相对路径均从项目根目录解释。

`--smoke` 与 `--overfit-samples` 互斥，检查结束后只打印并返回，不能写入或覆盖正式配置快照、history、曲线和 checkpoint。Day 4 准备训练与日志，Day 5 接通保存后运行正式 5 epochs；正式运行前重新固定种子并创建模型和优化器。具体规则见[运行模式说明](../../../docs/week2.md#run-modes)。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `mnist_split.json` | Day 2：划分索引与 seed |
| `figures/mnist_batch.png` | Day 2：训练批次中前 8 张图片及真实标签 |
| `environment.txt / conda-*.txt / pip-freeze.txt` | 实际环境检查与安装后快照 |
| `config.yaml` | Day 5：正式运行使用的配置快照 |
| `history.csv / figures/training_curves.png` | Day 5：训练/验证指标 |
| `test_metrics.json / figures/mistakes.png` | Day 6：冻结配置后的独立测试 |
| `outputs/mnist_mlp/{best,last}.pt` | 权重与配置，保存于项目根 outputs 下 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

