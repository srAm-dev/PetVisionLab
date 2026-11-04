# 第三周：CNN、ResNet 与 BN / 残差对照

学习指南：[第三周逐日任务](../../../docs/week3.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="progress"></a>
## 当前进度（2026-11-04）

Day 1 第二步的数据加载接口迁移已完成：build_mnist_loaders 接收外部创建的 train_transform / eval_transform，训练与验证使用独立 Dataset；train_cv.py、evaluate.py、inspect_mnist.py、check_mnist_training.py 四处调用已同步修改。MNIST 保留原来的 ToTensor → Normalize，归一化仍从原配置读取。操作说明见[统一接口步骤](../../../docs/week3.md#unified-loader-interface)。

CIFAR 的 build_cifar10_loaders 已完成实现，模拟数据与真实数据检查均通过。inspect_cifar10.py 已接入加载函数、子集数量 / 批次检查、验证变换确定性检查和图片保存。CNN、BasicBlock、ResNet18 和残差开关检查已完成；四组均已有完整 100 轮记录，Day 5 的产物核验与观察见下方。

### 接口迁移检查记录（2026-10-28）

- 迁移时执行的 2 项数据接口检查和原有 6 项训练检查均通过，共 8 项；数据检查覆盖训练 / 评估变换隔离及固定划分复用。
- train_cv.py 的 smoke 覆盖训练 / 验证各 256 条；check_mnist_training.py 在固定 64 条训练样本上达到 100% 准确率。
- CPU 重载原 MNIST best，完整验证结果为 n_samples=5000、accuracy=0.9642、loss=0.11792092759907245，与此前 CPU 复核一致。
- inspect_mnist.py 的形状、类型和像素范围检查通过；检查用图片重定向到临时目录，未覆盖原展示图。本次未重跑正式五轮训练或测试集评估。

这些结果用于核对接口迁移，没有产生 CIFAR-10 的训练或评估指标。

### CIFAR 加载函数检查（2026-10-28）

代码审查和运行检查未发现 build_cifar10_loaders 的功能错误，本次未修改函数。使用模拟 CIFAR10 Dataset、合成 RGB 图片和临时划分文件调用实际加载函数，并使用 inspect_cifar10.py 中定义的两套 v2 变换，检查通过：

- 默认返回 45000 / 5000 / 10000，训练与验证索引不相交且完整覆盖官方训练集的索引范围；训练随机取样，验证 / 测试顺序取样，均保留尾批。
- 三个 loader 输出 `[64,3,32,32]` 的 float32 图片与 `[64]` 的 long 标签，像素范围为 [-1,1]，标签在 0..9。
- 训练与验证底层 Dataset 独立；改变训练预处理不影响验证 / 测试，同一验证样本重复读取一致。
- 重复调用复用原划分且不改写文件；相同 seed 的新 loader 具有相同取样顺序；validation_size=10000 时返回 40000 / 10000 / 10000。
- 6 种非法 validation_size、已有文件的 seed / 大小冲突，以及 16 种损坏划分均被拒绝，已有文件未被覆盖。损坏案例覆盖元数据、索引类型、数量、范围、组内重复及组间重叠。

另运行现有 `tests/test_datasets.py` 与 `tests/test_cv_training.py`，共 **8 passed**；其中现有数据测试针对 MNIST，CIFAR 检查是当次临时运行，不计入这 8 项。该阶段使用模拟数据，未写正式划分、训练指标或图片；之后的真实数据检查见下节。

### 真实数据检查（2026-10-28）

运行 `python scripts/inspect_cifar10.py`，真实数据读取与脚本断言通过：

- 子集数量为 45000 / 5000 / 10000。
- 图片为 `[64,3,32,32]`、float32，像素范围 [-1,1]；标签为 `[64]`、int64，范围 0..9。
- 训练 / 验证底层 Dataset 独立，同一验证图片重复读取一致。
- 已生成 [cifar10_split.json](cifar10_split.json) 与 [cifar10_batch.png](figures/cifar10_batch.png)。划分 seed=42，文件 SHA-256 为 `482ec710828a8da2f0fefda6091dc38c1bbbed0505a09325d2989fb5db4a2d27`。

检查仅涉及数据与展示，未训练分类模型或计算测试集指标。

### Day 2 / Day 4 基线记录核对（2026-11-01）

以下数值读取已有 history、best / last 和 run_summary；两组都已完成 100 轮，initial.pt 的 step 均为 0。未重新训练 on 组。

| variant | batch | 已完成 epochs | best epoch | best val accuracy | 运行材料 |
| --- | --- | --- | --- | --- | --- |
| cnn_bn_on | 2048 | 100 | 93 | 77.24% | [配置](runs/cnn_bn_on/config.yaml)、[history](runs/cnn_bn_on/history.csv)、[summary](runs/cnn_bn_on/run_summary.json) |
| resnet_on | 512 | 100 | 80 | 91.02% | [配置](runs/resnet_on/config.yaml)、[history](runs/resnet_on/history.csv)、[summary](runs/resnet_on/run_summary.json) |

### Day 5 初始状态与运行检查（2026-11-01）

- `train_cv.py::main` 已接入 training.init_checkpoint，在创建优化器之前检查 step、seed、模型结构、状态键、shape 和 dtype，复制后逐项 torch.equal。CNN off 的 8 项共有状态与 on 完全一致，来源独有的 15 项均属于三个 BN；ResNet off 的 127 项状态与 on 完全一致，两组最大共有状态差均为 0。
- ResNet 配置统一使用 model.use_residual；省略时按 True 处理。evaluate.py 已兼容旧 checkpoint 未写该字段的情况，仍拒绝开启 / 关闭不匹配。
- 现有 tests 目录检查为 **40 passed**，覆盖初始状态对齐、新优化器、异常初始状态拒绝、输出保护和旧配置加载；未增加测试文件。
- 两组真实 GPU smoke 均通过：CNN off 的训练 / 验证各处理 4096 条，ResNet off 各处理 1024 条，loss / accuracy 有限。smoke 只用于接线检查，正式每轮验证覆盖 5000 条。
- 两份 off 配置保留对应 on 的数据、增强、优化器和 100 轮预算：CNN batch=2048，ResNet batch=512。四份集中快照见 [CNN on](configs/cnn_cifar10.yaml)、[CNN off](configs/cnn_no_bn_cifar10.yaml)、[ResNet on](configs/resnet_cifar10.yaml)、[ResNet off](configs/resnet_no_residual_cifar10.yaml)；on 快照来自原运行目录。
- 两组 off 现已完成正式训练，产物与来源初始状态的再次核对见下节。

### Day 5 正式训练验收（2026-11-04）

本次读取四组已有训练产物，并运行 `python -m pytest tests -q`，结果为 **40 passed**；本次检查未启动正式训练。对照问题是：在固定模型主分支、初始权重和训练预算下，BN 与残差连接分别怎样影响拟合速度和验证表现。

| variant | 已完成 epochs | 总参数量 | best epoch | best val loss | best val accuracy |
| --- | --- | --- | --- | --- | --- |
| cnn_bn_on | 100 | 94,986 | 93 | 0.682113 | 77.24% |
| cnn_bn_off | 100 | 94,538 | 99 | 1.112517 | 61.50% |
| resnet_on | 100 | 11,184,650 | 80 | 0.451596 | 91.02% |
| resnet_off | 100 | 11,184,650 | 84 | 0.446547 | 90.28% |

表中 loss / accuracy 取自同一个 best epoch，并与各组 best.pt 的保存值逐项核对；split=val，验证集为固定划分中的 5000 条。最佳模型按验证 accuracy 选择，持平时保留较早一轮，表中 loss 不表示该列全程最小值。

| variant | 实际配置 | 逐轮指标 | 资源记录 | 曲线 |
| --- | --- | --- | --- | --- |
| cnn_bn_on | [config](runs/cnn_bn_on/config.yaml) | [history](runs/cnn_bn_on/history.csv) | [summary](runs/cnn_bn_on/run_summary.json) | [曲线](runs/cnn_bn_on/figures/training_curves.png) |
| cnn_bn_off | [config](runs/cnn_bn_off/config.yaml) | [history](runs/cnn_bn_off/history.csv) | [summary](runs/cnn_bn_off/run_summary.json) | [曲线](runs/cnn_bn_off/figures/training_curves.png) |
| resnet_on | [config](runs/resnet_on/config.yaml) | [history](runs/resnet_on/history.csv) | [summary](runs/resnet_on/run_summary.json) | [曲线](runs/resnet_on/figures/training_curves.png) |
| resnet_off | [config](runs/resnet_off/config.yaml) | [history](runs/resnet_off/history.csv) | [summary](runs/resnet_off/run_summary.json) | [曲线](runs/resnet_off/figures/training_curves.png) |

核验结果：

- 四份 history 均只有五列表头和 100 行数据，epoch 从 1 到 100 连续；loss / accuracy 均有限，best 对应首次最高验证 accuracy 的行，last 对应第 100 轮，保存的指标与各自行一致。
- 四组 initial.pt 均为 step=0；CNN 两组 8 项共有状态完全相同，差异仅为 on 组的 15 项 BN 状态。ResNet 两组 127 项状态完全相同，off 组的 6 项投影 weight / bias 在训练后仍与初始值相同，符合关闭时跳过投影计算的设计。
- 两对各自的数据与训练预算相同：固定 45000 / 5000 划分、seed=42、100 epochs、AdamW lr=1e-3、weight_decay=1e-4；CNN batch=2048，ResNet batch=512。四个运行目录和 checkpoint 目录互不相同，集中配置与各运行快照一致，模型状态均可严格加载。
- CNN off 移除了 BN 的 448 个可学习仿射参数；ResNet off 保留投影模块，因此两组总参数量相同，但 off 跳过投影的实际计算。

训练现象：CNN 在第 50 轮时，BN on / off 的 train accuracy 分别为 73.44% / 53.23%，val accuracy 为 64.62% / 53.36%；本次固定预算下，BN on 拟合更快，最佳验证准确率高 15.74 个百分点。BN on 的验证曲线仍有波动，收敛速度、验证指标与曲线稳定性应分别描述。

ResNet 在第 50 轮时，残差 on / off 的 val accuracy 为 88.98% / 87.58%，最终最佳验证准确率相差 0.74 个百分点。两组后期训练 loss 继续下降，验证 loss 未同步下降，应按已经冻结的 best 选择规则比较。关闭残差后仍能在本预算内训练到 90.28% 的最佳验证准确率。

以上结论限定于当前单种子、模型和预算；两组的增强策略与 seed 一致，未逐样本核对随机增强决策，也未据训练总耗时推断单独某层的加速效果。独立重载完整验证、两张合并对照图、results.csv、误分类分析和周总结继续按 Day 6 完成。

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
| Day 2–4 | `python -m pytest tests/test_resnet.py -q` | CNN、BasicBlock、ResNet18 前反向与阶段形状检查 |
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

数据读取由 `src/petvision/data/datasets.py::build_cifar10_loaders` 返回 45k/5k/10k 的 train / val / test loader；`scripts/inspect_cifar10.py::main` 创建两个 transform 后传入该函数，打印子集数量并检查批次类型和形状。具体签名与固定划分步骤见[Day 1 数据步骤](../../../docs/week3.md#cifar-data)。

配置中的 validation_size 和 num_workers 传给 build_cifar10_loaders，归一化 / 增强参数用于创建 `torchvision.transforms.v2` 变换；正式划分固定为 45000/5000。训练采用 ToImage → RandomCrop → RandomHorizontalFlip → ToDtype(torch.float32,scale=True) → Normalize；验证和测试去掉随机裁剪与翻转；`scripts/train_cv.py::main` 和 `scripts/evaluate.py::main` 的 resnet18 分支在 ToDtype 之前增加 Resize(96,96)，CNN 分支保留 32×32。配置中的 data.image_size 记录原图尺寸 32。完整示例与数值范围见 [v2 预处理步骤](../../../docs/week3.md#cifar-v2-transforms)。逐轮追加 history、耗时与峰值显存保存到 run_summary.json，具体见[资源记录步骤](../../../docs/week3.md#training-resources)。

初始状态按[训练前保存步骤](../../../docs/week3.md#initial-state)产生：Day 2 写 `outputs/cnn_bn_on/initial.pt`，Day 4 写 `outputs/resnet_on/initial.pt`，均在正式训练的第一次前向前保存，step=0。smoke/overfit 的模型不用于这些文件。

Day 5 在 BN off 配置设置 `training.init_checkpoint: outputs/cnn_bn_on/initial.pt`，在 residual off 配置设置 `training.init_checkpoint: outputs/resnet_on/initial.pt`。这两个字段在 `scripts/train_cv.py::main` 的模型构造之后、optimizer 创建之前读取：先检查来源 step=0，逐 key 核对 shape 并复制；残差对照要求模型状态键完全相同，BN 对照只允许预期的 BN 键差异。复制后由 run_training 保存各自 initial.pt，再从新优化器、零进度开始训练；best/last 用于训练结果和评估。文件不存在时先补齐该对基线，不能替换成训练后权重。运行命令不变，这个字段由配置读取。数据顺序使用独立 Generator；相同 seed 只能作为随机设置记录，若未独立控制随机增强，不声称逐样本增强完全一致。

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

