# 第三周：CNN、ResNet 与 BN / 残差对照

学习指南：[第三周逐日任务](../../../docs/week3.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="progress"></a>
## 当前进度（2026-10-28）

Day 1 第二步的数据加载接口迁移已完成：build_mnist_loaders 接收外部创建的 train_transform / eval_transform，训练与验证使用独立 Dataset；train_cv.py、evaluate.py、inspect_mnist.py、check_mnist_training.py 四处调用已同步修改。MNIST 保留原来的 ToTensor → Normalize，归一化仍从原配置读取。操作说明见[统一接口步骤](../../../docs/week3.md#unified-loader-interface)。

CIFAR 的 build_cifar10_loaders 已完成实现，模拟数据与真实数据检查均通过。inspect_cifar10.py 已接入加载函数、子集数量 / 批次检查、验证变换确定性检查和图片保存。后续继续按指南完成卷积形状练习与 CNN / ResNet 实现，四组对照实验尚未开始。

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

配置中的 validation_size 和 num_workers 传给 build_cifar10_loaders，归一化 / 增强参数用于创建 `torchvision.transforms.v2` 变换；正式划分固定为 45000/5000。训练采用 ToImage → RandomCrop → RandomHorizontalFlip → ToDtype(torch.float32,scale=True) → Normalize；验证和测试去掉随机裁剪与翻转。完整示例与数值范围见 [v2 预处理步骤](../../../docs/week3.md#cifar-v2-transforms)。逐轮追加 history、耗时与峰值显存保存到 run_summary.json，具体见[资源记录步骤](../../../docs/week3.md#training-resources)。

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

