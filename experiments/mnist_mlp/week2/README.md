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

以下命令均从项目根目录执行，先激活对应 Conda 环境。当前已有 Day1–4 的实现，以及 Day5 的 checkpoint 保存加载和独立 evaluate 入口；相关检查已通过。含 best/last 的正式五轮训练已完成，独立加载验证也已通过；最终测试和结果输出按 Day6 继续完成。若要重新训练，先归档需要保留的当前结果与权重。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/check_environment.py --output-dir experiments/mnist_mlp/week2` | 打印并记录环境 |
| Day 1 | `python scripts/torch_basics.py` | Tensor 与 autograd 检查 |
| Day 2 | `python scripts/inspect_mnist.py` | 固定划分与数据检查 |
| Day 3–6 | `python -m pytest tests/test_cv_training.py -q` | 当前 6 项：形状 / 梯度 / 设备 / 统计、保存加载、best/last 选择 |
| Day 4 第三步（按指南创建脚本后） | `python scripts/check_mnist_training.py` | 独立运行两批 smoke 与固定 64 样本拟合 |
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

手写三分类交叉熵检查：正确类 logit 从 0 提高到 2 时，loss 从约 1.098612 降至 0.239545。Day 3 检查使用固定尺寸的 `MLP()`；参数化改造按 Day 4 第四步完成，并保留该默认调用。

Day 4 第一步检查记录（2026-10-25）：set_seed 的 Python / NumPy / Torch 随机序列复现检查通过。训练 / 验证函数通过 10 项临时小样本检查，覆盖 CPU/CUDA、非整批的按样本加权损失、部分遍历的实际样本数、训练前清梯度与参数更新、eval/no_grad 下参数和已有梯度不变，以及空输入报错。原有 Day 3 三项测试也通过。另在 `cuda:0` 上取真实 MNIST 训练和验证各一批，每批 128 条，确认训练更新参数、验证不改变参数或已有梯度；这属于批次检查，尚未进行整轮正式训练或完整验证。

Day 4 第二、三步示例验证（2026-10-25）：统计检查覆盖两批大小为 2/1 的已知数据及只读取首批的情况，均符合全体样本平均损失与实际样本数。MNIST smoke 的训练、验证各处理两批、每批 128 条；固定 64 条训练样本更新 200 次后，loss 约为 0.000319，训练 accuracy=100%。这是小样本拟合检查，未评估完整验证或测试集，也未保存正式训练结果。

Day 4 第四步接口检查（2026-10-25）：现有 4 项测试与 6 项临时接口检查共 10 项通过。MLP 默认结构仍为 235146 个参数，自定义 4→8→4→3 结构为 91 个参数，CPU/CUDA 前反向正常。数据函数默认读取 55k/5k/10k；独立临时划分使用 validation_size=10000 时为 50k/10k/10k，索引无重叠且完整覆盖官方训练集。mean/std、num_workers 和 batch_size 的传入值生效；旧划分数量不匹配时明确报错，原划分文件保持不变。当时实际 YAML 可创建模型与数据并完成一个批次的前反向；配置驱动的 smoke 检查随后在“前六步检查”中通过，当时正式训练仍待进行。

MLP 循环组装检查（2026-10-25）：现有 4 项测试和 3 项临时回归检查共 7 项通过，覆盖 CPU/CUDA 的一层、两层、三层隐藏层配置及前反向；空隐藏层序列明确报错。默认参数量仍为 235146，默认 state_dict 键名与原两隐藏层实现一致，相同权重时输出一致。正式 MNIST 配置仍为 hidden_dims=[256,128]。

Day 4 前六步检查（2026-10-25）：现有 4 项测试与 8 项临时入口检查共 12 项通过，包含命令行互斥 / 正数校验、不支持的模型 / 数据集 / 优化器 / dtype 明确报错，以及过拟合样本数上限校验。实际运行 train_cv.py 的 --smoke，训练和验证各处理 256 条样本；--overfit-samples 64 完成 200 次更新后，loss 约为 0.000319，固定训练样本 accuracy=100%。使用 RTX 4060 Laptop GPU。当时第七步尚未完成，该次检查未执行完整正式训练。

Day4 文件核对（2026-10-25）：保存与绘图已实现，`history.csv` 有 epoch=1..5 的五行记录，与配置的 epochs=5 对应。第 5 轮 train_loss≈0.085465、train_accuracy≈97.30%，val_loss≈0.117921、val_accuracy=96.42%；这也是该表验证准确率最高的一轮。以上来自已有 CSV，本次文档检查没有重新训练。当时没有 best.pt/last.pt，无法据此重建当时模型或验证保存加载一致性，也没有独立测试结果。这份阶段记录现已归档，Day5 的配套权重与日志见下方新记录。

Day4 项目检查（2026-10-25）：`python -m pytest tests/test_cv_training.py -q -s` 的现有 4 项测试通过，设备检查使用 cuda:0。前面分步骤记录描述各自检查当时的范围，以现有文件和下方 Day5 检查判断当前进度。

当前入口另用临时合成数据验证了三个模式：正式分支生成两轮 CSV、配置快照和有效曲线；smoke 与 overfit 均保持已有结果文件的内容和修改时间不变。这是入口与文件隔离检查，输出保存在临时目录，不作为 MNIST 性能指标。

Day5 检查（2026-10-26）：已修正 checkpoint 的 CPU 加载映射、evaluate.py 的入口保护、设备读取顺序和配置键名；评估核对模型、归一化、划分数量、种子及实际划分 hash，允许更改数据缓存位置、评估 batch 与设备。现有 6 项测试和 9 项临时评估检查共 15 项通过。保存加载后固定输入的 logits 最大绝对差为 0；模拟验证准确率 `[0.80,0.85,0.82,0.85]` 时，实际训练入口保存的 best 为第 2 轮、last 为第 4 轮，持平不覆盖较早 best。

真实 MNIST 加载检查：使用一个训练批次更新后的临时模型保存 checkpoint，再在独立进程分别用 CUDA / batch=128 和 CPU / batch=257 完整评估 5000 条验证数据。准确率一致，CUDA loss 差为 0，CPU loss 差约为 `5.0e-8`；同一 CUDA 批次重载 logits 差为 0。临时 checkpoint 不作为正式训练结果，没有评估官方测试集。

当前进度：旧五轮记录已移动到 [Day4 归档配置](archive/day4_without_checkpoint/config.yaml)、[归档指标](archive/day4_without_checkpoint/history.csv)和[归档曲线](archive/day4_without_checkpoint/training_curves.png)。Day5 正式五轮训练及对应 best/last 已生成，独立验证通过；接下来进入 Day6。

Day5 复查（2026-10-27）：现有 6 项测试全部通过，设备测试使用 cuda:0。核对正式配置快照与训练 YAML 一致，history 包含 epoch=1..5；best/last 的配置、seed、划分 SHA-256、epoch 与验证指标均和对应日志一致。第 5 轮验证准确率最高，故 best 与 last 都为第 5 轮，模型参数相同，属于正常结果。

本次只检查已有训练产物，没有重新训练。独立运行上表的 `evaluate.py ... --split val` 命令，在 GPU 上加载 best 并完整评估 5000 条验证样本，loss=0.1179209182560444、accuracy=0.9642，与第 5 轮日志完全一致。正式权重在 CPU 上经临时文件再次保存加载，固定随机输入的 logits 最大绝对差为 0；这是序列化一致性检查，不是额外训练结果。尚未评估官方测试集。

Day 4 的配置接线顺序：先按[第四步](../../../docs/week2.md#mnist-configurable-interfaces)改造 MLP 和数据接口，验证默认值兼容及自定义参数生效，再写 YAML；第五步读取成字典，第六步将字段传给构造函数 / 数据函数。MLP 仍只接收 self 时不能使用 MLP(input_dim=...)；数据函数也需要先支持 validation_size、num_workers、mean、std。接口接通后再运行表中的 train_cv.py 检查模式。

完整 YAML 见[Day 4 配置示例](../../../docs/week2.md#mnist-config)，参数解析见[最小入口示例](../../../docs/week2.md#cli-config)。统一使用 `data.root`、`data.split_path`、`data.batch_size` 和 `training.learning_rate`；相对路径均从项目根目录解释。

第七步的保存 / 绘图函数和正式 else 分支已接入 train_cv.py：每轮收集 history，循环结束后一次写 CSV 和曲线。完整示例见[结果管理步骤](../../../docs/week2.md#run-modes)。当前 if / elif / else 已隔离检查模式与正式输出；指南中的显式 return 可用于明确提前结束 main。

`--smoke` 与 `--overfit-samples` 互斥，检查结束后只打印并返回，不能写入或覆盖正式配置快照、history、曲线和 checkpoint。Day 4 准备训练与日志，Day 5 接通保存后运行正式 5 epochs；正式运行前重新固定种子并创建模型和优化器。具体规则见[运行模式说明](../../../docs/week2.md#run-modes)。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `mnist_split.json` | Day 2：划分索引与 seed |
| `figures/mnist_batch.png` | Day 2：训练批次中前 8 张图片及真实标签 |
| `environment.txt / conda-*.txt / pip-freeze.txt` | 实际环境检查与安装后快照 |
| [归档 config.yaml](archive/day4_without_checkpoint/config.yaml) | Day4 五轮运行配置，未配套权重 |
| [config.yaml](config.yaml) | Day5 正式五轮运行配置快照 |
| [归档 history.csv](archive/day4_without_checkpoint/history.csv) / [归档曲线](archive/day4_without_checkpoint/training_curves.png) | Day4 五轮训练 / 验证记录 |
| [history.csv](history.csv) / [训练曲线](figures/training_curves.png) | Day5 正式五轮指标与曲线，与该次 best/last 配套 |
| `test_metrics.json / figures/mistakes.png` | Day 6：冻结配置后的独立测试 |
| `outputs/mnist_mlp/{best,last}.pt` | 权重与配置，保存于项目根 outputs 下 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

