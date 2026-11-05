# 第三周：CNN、ResNet 与 BN / 残差对照

学习指南：[第三周逐日任务](../../../docs/week3.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

这里记录 CIFAR-10 四组实验的环境、命令和验证结果。首次跟做见第三周指南；复验时使用各组保存的配置和对应权重。

<a id="progress"></a>

## 当前进度（2026-11-04）

Day 1 第二步的数据加载接口迁移已完成：build_mnist_loaders 接收外部创建的 train_transform / eval_transform，训练与验证使用独立 Dataset；train_cv.py、evaluate.py、inspect_mnist.py、check_mnist_training.py 四处调用已同步修改。MNIST 保留原来的 ToTensor → Normalize，归一化仍从原配置读取。操作说明见[统一接口步骤](../../../docs/week3.md#unified-loader-interface)。

CIFAR 的 build_cifar10_loaders 已完成实现，模拟数据与真实数据检查均通过。inspect_cifar10.py 已接入加载函数、子集数量 / 批次检查、验证变换确定性检查和图片保存。CNN、BasicBlock、ResNet18 和残差开关检查已完成；四组均已有完整 100 轮记录。Day 6 的四组 CUDA 独立验证、results.csv、对照曲线、验证错图、[重写草稿运行检查](#basicblock-rewrite)和[第五周交接清单](#week5-handoff)均已记录，[周总结](#weekly-summary)已补齐；独立重写与解释能力由本人自查。

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

另运行现有 `tests/test_datasets.py` 与 `tests/test_cv_training.py`，共 8 passed；其中现有数据测试针对 MNIST，CIFAR 检查是当次临时运行，不计入这 8 项。该阶段使用模拟数据，未写正式划分、训练指标或图片；之后的真实数据检查见下节。

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
- 现有 tests 目录检查为 40 passed，覆盖初始状态对齐、新优化器、异常初始状态拒绝、输出保护和旧配置加载；未增加测试文件。
- 两组真实 GPU smoke 均通过：CNN off 的训练 / 验证各处理 4096 条，ResNet off 各处理 1024 条，loss / accuracy 有限。smoke 只用于检查数据与训练调用，正式每轮验证覆盖 5000 条。
- 两份 off 配置保留对应 on 的数据、增强、优化器和 100 轮预算：CNN batch=2048，ResNet batch=512。四份集中快照见 [CNN on](configs/cnn_cifar10.yaml)、[CNN off](configs/cnn_no_bn_cifar10.yaml)、[ResNet on](configs/resnet_cifar10.yaml)、[ResNet off](configs/resnet_no_residual_cifar10.yaml)；on 快照来自原运行目录。
- 两组 off 现已完成正式训练，产物与来源初始状态的再次核对见下节。

### Day 5 正式训练验收（2026-11-04）

本次读取四组已有训练产物，并运行 `python -m pytest tests -q`，结果为 40 passed；本次检查未启动正式训练。对照问题是：在固定模型主分支、初始权重和训练预算下，BN 与残差连接分别怎样影响拟合速度和验证表现。

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

以上结论限定于当前单种子、模型和预算；两组的增强策略与 seed 一致，未逐样本核对随机增强决策，也未据训练总耗时推断单独某层的加速效果。独立重载完整验证、results.csv、两张合并对照图、误分类分析及第五周材料交接已按下面 Day 6 对应步骤完成；BasicBlock 重写草稿的运行记录见第三步，关键结论与后续巩固内容见[周总结](#weekly-summary)。

### Day 6 第一步：独立验证与汇总表（2026-11-04）

已在四个独立进程中通过 `scripts/evaluate.py` 的评估入口加载各组 best.pt，使用 CUDA、FP32 和 `--split val` 完整评估。每组实际处理 5000 条验证样本，loss 与 accuracy 均与 best 所在 epoch 的 history 和 checkpoint 保存值完全一致，差值均为 0。

| variant | 加载 epoch | 样本数 | 实际输入边长 | 独立验证 loss | 独立验证 accuracy | 原始验证记录 |
| --- | --- | --- | --- | --- | --- | --- |
| cnn_bn_on | 93 | 5000 | 32 | 0.682113 | 77.24% | [val_metrics.json](runs/cnn_bn_on/val_metrics.json) |
| cnn_bn_off | 99 | 5000 | 32 | 1.112517 | 61.50% | [val_metrics.json](runs/cnn_bn_off/val_metrics.json) |
| resnet_on | 80 | 5000 | 96 | 0.451596 | 91.02% | [val_metrics.json](runs/resnet_on/val_metrics.json) |
| resnet_off | 84 | 5000 | 96 | 0.446547 | 90.28% | [val_metrics.json](runs/resnet_off/val_metrics.json) |

结果表为 [results.csv](results.csv)，只有一行表头与四行实验记录。val_loss / val_accuracy 取自同一个 best epoch 的历史行；params / train_seconds 读取各组 run_summary.json，epochs / batch 读取各组实际 config.yaml，并保存配置、checkpoint 和独立验证记录的来源路径。

实际输入尺寸由验证时模型接收的 images.shape 核对，已将 image_size=32 / 96 补入相应 run_summary.json；配置的 data.image_size 继续表示原始 CIFAR-10 图片边长 32。每份 val_metrics.json 还保存设备、精度、PyTorch 版本、checkpoint 与划分 SHA-256，以及独立验证与历史指标的差值。现有 evaluate.py 打印验证指标，本次按第一步的草稿流程将这些指标另存为 JSON 并生成 CSV。

已回读 CSV、JSON 和运行摘要，核对四个 variant、5000 条样本、best 轮次、指标与来源路径；checkpoint、history、训练配置和固定划分的前后 hash 一致。本次只评估已有模型。

### Day 6 第二步：对照曲线与验证误分类（2026-11-04）

对照图直接读取四份 history.csv 中的 100 轮真实记录；每图包含 loss、accuracy 两个子图，同一 variant 使用同一颜色，实线表示 train、虚线表示 val，accuracy 以百分比显示。

- [BN on/off 对照](figures/bn_comparison.png)：CNN 的卷积结构、共有初始参数与预算相同，比较 BN 开关。
- [残差 on/off 对照](figures/residual_comparison.png)：ResNet18 的主分支深宽、初始状态与预算相同，比较旁路计算和相加的开关。

曲线观察：BN on 在本预算下拟合更快、验证准确率更高，验证曲线仍有波动；BN off 的训练与验证准确率都在约 60%，当前训练集拟合程度较低。ResNet on 的训练准确率首次达到 90% / 98% 分别在第 17 / 55 轮，off 为第 23 / 68 轮，体现了达到相同拟合程度所需 epoch 的差异；100 轮后的最佳验证准确率相差 0.74 个百分点。两组 ResNet 后期训练 loss 持续下降，而验证 loss 未同步下降，比较时仍按既定规则使用 best。

`scripts/evaluate.py::main` 已将 collect_mistakes 和绘图接到 val / test 两种 split，完整指标先统计，再收集最多 8 张错图；文件名分别为 mistakes_val.png / mistakes_test.png。CIFAR 图片按配置的三通道 mean/std 逆归一化并转为 HWC，标题显示实际类别名和预测类别的 softmax 得分，整张图标出 variant、split 与 checkpoint epoch。没有错图时只移除当前 split 的旧图片。

本次四组都使用 CUDA 和各自 best 权重，仅运行验证集。每组完整评估 5000 条，指标与第一步的 val_metrics.json 完全一致；下表错分总数由完整验证的样本数与 accuracy 核算，图片展示的是按固定验证顺序最先遇到的 8 个错误。

| variant | 完整验证集错分数 | 展示数量 | 错图 |
| --- | --- | --- | --- |
| cnn_bn_on | 1138 | 8 | [mistakes_val.png](runs/cnn_bn_on/figures/mistakes_val.png) |
| cnn_bn_off | 1925 | 8 | [mistakes_val.png](runs/cnn_bn_off/figures/mistakes_val.png) |
| resnet_on | 449 | 8 | [mistakes_val.png](runs/resnet_on/figures/mistakes_val.png) |
| resnet_off | 486 | 8 | [mistakes_val.png](runs/resnet_off/figures/mistakes_val.png) |

样例观察：CNN 两组均出现 automobile→truck、horse→deer、deer→horse；ResNet on 中出现 cat→dog，得分可达 99.03%，ResNet off 中出现 dog→cat，得分可达 99.96%。这些图中还包含低分辨率、细节不清和背景较复杂的样本；外观相似、细节不足可能与错误有关，具体原因仍需进一步验证。展示 ResNet 输入时使用的是原始 32×32 图片 Resize 后的 96×96 张量。

这 8 张样例用于观察具体错误，不能据此估计全体验证集的类别混淆频率；单个错误的高 softmax 得分也不证明其判断正确。单种子、增强随机决策未逐样本对齐，以及 residual off 跳过投影计算的限制继续适用。

本次 `python -m pytest tests -q` 为 46 passed，在已有 test_cv_training.py 中检查了 MNIST / CIFAR、val / test 的图像还原与命名、8 张展示与完整样本统计的边界，以及旧错图清理。已确认本次生成图表后，checkpoint、history、配置、运行摘要、第一步验证记录和 results.csv 的 hash 均未改变。

<a id="basicblock-rewrite"></a>

### Day 6 第三步：重写草稿运行检查（2026-11-04）

草稿为 [tmp/basic_block_rewrite.py](../../../tmp/basic_block_rewrite.py)。保留本文件自己定义的 BasicBlock，新增 check_block 和 main：实际调用重写类完成前向与反向，原实现以 ReferenceBasicBlock 导入，仅用于相同输入、相同权重的数值对照。

在项目根目录激活 petvision 环境后运行：

```bash
python tmp/basic_block_rewrite.py
```

本次实际使用 CUDA、FP32，seed=42，PyTorch 2.14.1+cu130，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。脚本会自动选择可用的 CUDA，否则使用 CPU，并在开头打印设备和版本。两种块均设置 use_residual=True，输入是随机特征图 `[2,64,32,32]`。

检查顺序：先严格复制草稿块权重到参考块，并让两者都处于 train 模式；在草稿的 bn2 和存在的 shortcut 上注册 hook，记录一次真实 forward 的两条路径形状，结束后移除 hook；核对两条路径和最终输出均符合预期。随后对 `output.square().mean()` 调用 backward，逐项检查输入及全部活跃参数的梯度存在且有限，并用 torch.testing.assert_close 对照两份实现的输出、输入梯度和参数梯度。

| 检查项 | 例 A：普通块 | 例 B：下采样块 |
| --- | --- | --- |
| 构造参数 | in=64、out=64、stride=1、use_1x1conv=False | in=64、out=128、stride=2、use_1x1conv=True |
| 主分支出口 | `[2,64,32,32]` | `[2,128,16,16]` |
| 旁路出口 | `[2,64,32,32]`，原输入 | `[2,128,16,16]`，1×1 Conv 投影 |
| 最终输出 | `[2,64,32,32]` | `[2,128,16,16]` |
| 标量目标值 | 0.99026048 | 0.67734408 |
| 有限梯度的参数张量数 | 8 | 10，含 shortcut.weight / bias |
| 输入梯度范数 | 0.00697449 | 0.00594122 |
| 与原实现最大输出差 | 0 | 0 |
| 与原实现最大输入梯度差 | 0 | 1.09139364e-11 |
| 与原实现最大参数梯度差 | 0 | 1.25495717e-07 |

两例均通过断言，进程退出码为 0；FP32 梯度对照按 torch.testing.assert_close 的默认容差检查，表中的微小差异在容差内。8 / 10 统计的是参数张量数量，非参数元素总数。此检查使用小张量构造计算图并反向，没有执行优化器更新。

结构解释：主分支依次为 3×3 Conv → BN → ReLU → 3×3 Conv → BN，再与恒等或 1×1 投影旁路相加，最后 ReLU，得到 `ReLU(F(x) + shortcut(x))`。提前对第二层 BN 的输出做 ReLU 会改变参与相加的 F(x)。反向传播时，在最终 ReLU 允许的位置，梯度既可经过主分支，也可经过旁路返回输入；恒等旁路直接传回，投影旁路则经过 1×1 Conv。能否有限参考独立写出并解释这段流程，仍由本人自查。

本次脚本 SHA-256 为 `8bd798a63229f0167243a4602e206d1414302877f808e210bf0c6a03bf211427`。本节保存对应运行条件、输入输出和结果；复现时同时保留此草稿并核对版本。原模型源码及 56 个既有训练 / 验证相关文件的 hash 均未改变。

<a id="week5-handoff"></a>

### Day 6 第四步：第五周材料交接（2026-11-04）

第五周从下表取得各组的实际运行材料。[results.csv](results.csv) 是四组验证结果汇总；下表的 history 和 summary 分别属于单组运行，不能互换。配置链接指向训练时保存在 runs 中的快照，四份[集中配置](configs/)与各自快照内容一致；后续复验优先使用这些快照。

| variant | 实际配置 | 逐轮指标 | 资源记录 | 交接权重 | 独立验证与权重校验值 |
| --- | --- | --- | --- | --- | --- |
| cnn_bn_on | [config.yaml](runs/cnn_bn_on/config.yaml) | [history.csv](runs/cnn_bn_on/history.csv) | [run_summary.json](runs/cnn_bn_on/run_summary.json) | [best.pt，epoch 93](../../../outputs/cnn_bn_on/best.pt) | [val_metrics.json](runs/cnn_bn_on/val_metrics.json) |
| cnn_bn_off | [config.yaml](runs/cnn_bn_off/config.yaml) | [history.csv](runs/cnn_bn_off/history.csv) | [run_summary.json](runs/cnn_bn_off/run_summary.json) | [best.pt，epoch 99](../../../outputs/cnn_bn_off/best.pt) | [val_metrics.json](runs/cnn_bn_off/val_metrics.json) |
| resnet_on | [config.yaml](runs/resnet_on/config.yaml) | [history.csv](runs/resnet_on/history.csv) | [run_summary.json](runs/resnet_on/run_summary.json) | [best.pt，epoch 80](../../../outputs/resnet_on/best.pt) | [val_metrics.json](runs/resnet_on/val_metrics.json) |
| resnet_off | [config.yaml](runs/resnet_off/config.yaml) | [history.csv](runs/resnet_off/history.csv) | [run_summary.json](runs/resnet_off/run_summary.json) | [best.pt，epoch 84](../../../outputs/resnet_off/best.pt) | [val_metrics.json](runs/resnet_off/val_metrics.json) |

已核对四个运行目录、输出目录及 checkpoint 保存的 config，组名和路径逐项对应；各组 initial / best / last 共 12 份状态均可严格加载。best 对应各自 history 中首次取得最高 val accuracy 的行，last 对应第 100 轮，initial 的 step=0。四份 val_metrics.json 中的 checkpoint_sha256 均与实际 best 文件一致，用该字段识别交接的具体权重版本。

两组 CNN 实际输入为 32×32，两组 ResNet 为 Resize 后的 96×96，配置里的 data.image_size 均记录原图尺寸 32。resnet_on 原始运行快照没有 use_residual 字段，按默认 True 重建；当前 configs/cv/resnet_cifar10.yaml 显式写 True，含义相同，保留原运行快照。其余三份入口配置与运行快照一致。

固定划分保存在 [cifar10_split.json](cifar10_split.json)：dataset=cifar10、version=1、seed=42，训练 / 验证索引分别为 45000 / 5000；已检查索引无重复、组间不相交且完整覆盖官方训练集。四组配置指向这一文件，best / last 和独立验证记录中的 split_sha256 均匹配：

```text
482ec710828a8da2f0fefda6091dc38c1bbbed0505a09325d2989fb5db4a2d27
```

第五周直接复用此文件及对应索引，不重新抽取验证集。BN 对照图为 [bn_comparison.png](figures/bn_comparison.png)，残差对照图为 [residual_comparison.png](figures/residual_comparison.png)；验证错误样例见第二步的四组链接。

本次复验的 Git 基准为 `193554b2ff5dc52405c2ea7cbe23c11c3d4cb029`。工作区包含尚未提交的 evaluate.py 修改，因此同时保留以下 SHA-256，标识交接时训练与评估相关源码的实际内容：

| 源码 | SHA-256 |
| --- | --- |
| [scripts/train_cv.py](../../../scripts/train_cv.py) | `98ecb54f7da0957e97b0000d7c3a530c079f35d6d0d784199fd0dc74df13c769` |
| [scripts/evaluate.py](../../../scripts/evaluate.py) | `a0921343f48f1be5428f07dbbe13dc3850bf2094fbf41ec55a89846d561cd92f` |
| [cnn.py](../../../src/petvision/models/cv/cnn.py) | `1d6198a0929237363dfe6f98bbc5a1205f4b6911a428f012566a869633e1192c` |
| [resnet.py](../../../src/petvision/models/cv/resnet.py) | `3829be21927bfff4d72b3b4296db5f96054a5ccbfd32bea8872d105513087202` |
| [datasets.py](../../../src/petvision/data/datasets.py) | `6d05c6ffbbe8f0cca85cc3958d48afee7ee39cad66a5d07aef8d551b06fc9cb6` |
| [cv_trainer.py](../../../src/petvision/training/cv_trainer.py) | `a9bebe1563d57d12aef064b799aed5e41bf4e3fe2839ad422972dab8876bb4b8` |
| [checkpoint.py](../../../src/petvision/utils/checkpoint.py) | `ce37bf4656fe0ec0dd1bb5d4ee1b84c7810fbdbc7ee879eba4de67e31307bdfb` |

复验环境为 Python 3.12.15、PyTorch 2.14.1+cu130、torchvision 0.29.1+cu130、CUDA runtime 13.0；设备为 NVIDIA GeForce RTX 4060 Laptop GPU，计算精度为 FP32。模型、配置、权重和数据划分的校验值应与上述材料一起保留。

本次在新进程中，使用上表 resnet_on 的原始配置重建模型、严格加载 best，并按原来的确定性预处理遍历完整验证集。从项目根目录、激活 petvision 环境后可复验：

```bash
python scripts/evaluate.py \
  --config experiments/cnn_vs_resnet/week3/runs/resnet_on/config.yaml \
  --checkpoint outputs/resnet_on/best.pt \
  --split val
```

实际结果为 epoch=80、n_samples=5000、loss=0.45159566712379456、accuracy=0.9102，模型收到的张量为 `[B,3,96,96]`、float32、CUDA；loss / accuracy 与第一步的独立验证、history 和 checkpoint 完全一致。其余三组的完整独立验证记录见表中 val_metrics.json，每组同样为 5000 条。

交接材料缺项：无。本次完成配置、日志、摘要、权重、划分、图表与源码的核对；复验前后 56 个相关文件的 hash 均未改变，测试集仍留到第五周统一评估。大权重和原始数据位于 Git 忽略目录 outputs / datasets/raw，跨机器复现时需一并复制对应目录，并按上面的 hash 校验划分与 best。Day 6 第三步的重写草稿运行检查已完成，本周完成情况和后续巩固内容见[周总结](#weekly-summary)。

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

以下命令均从项目根目录执行，先激活对应 Conda 环境。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。已有正式训练产物，复核结果时使用独立评估入口；需要重新训练时先归档原产物或选择新的运行与输出目录。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/inspect_cifar10.py` | 建立和检查固定划分 |
| Day 2-4 | `python -m pytest tests/test_resnet.py -q` | CNN、BasicBlock、ResNet18 前反向与阶段形状检查 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --smoke` | CNN 短运行 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --overfit-samples 64` | CNN 过拟合检查 |
| Day 2-5 | `python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml` | BN on |
| Day 5 | `python scripts/train_cv.py --config configs/cv/cnn_no_bn_cifar10.yaml` | BN off |
| Day 4-5 | `python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml` | 残差 on |
| Day 5 | `python scripts/train_cv.py --config configs/cv/resnet_no_residual_cifar10.yaml` | 残差 off |
| Day 6 | `python scripts/evaluate.py --config configs/cv/cnn_cifar10.yaml --checkpoint outputs/cnn_bn_on/best.pt --split val` | BN on 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/cnn_no_bn_cifar10.yaml --checkpoint outputs/cnn_bn_off/best.pt --split val` | BN off 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/resnet_cifar10.yaml --checkpoint outputs/resnet_on/best.pt --split val` | 残差 on 独立验证 |
| Day 6 | `python scripts/evaluate.py --config configs/cv/resnet_no_residual_cifar10.yaml --checkpoint outputs/resnet_off/best.pt --split val` | 残差 off 独立验证 |
| Day 6 | `python tmp/basic_block_rewrite.py` | 重写块的两种形状、梯度和原实现对照检查 |

## 配置与结果

BN 是批归一化，residual 是残差连接；on/off 表示启用 / 关闭。variant 是实验变体名称，路径中的 `<variant>` 要替换成 cnn_bn_on、cnn_bn_off、resnet_on 或 resnet_off，不保留尖括号。卷积、池化、stage 和初始状态对齐的含义见[本周术语与约定](../../../docs/week3.md#cnn-terms)。

本轮实际配置：seed=42、100 epochs、AdamW lr=1e-3、weight_decay=1e-4、FP32；CNN 两组 batch=2048，ResNet 两组 batch=512，两对实验内部预算一致。训练/验证为 45k/5k，增强只作用训练。各组 experiment_dir 为 `experiments/cnn_vs_resnet/week3/runs/<variant>`，四组 checkpoint 输出目录分别为 `outputs/cnn_bn_on`、`outputs/cnn_bn_off`、`outputs/resnet_on`、`outputs/resnet_off`，best 按验证 accuracy 选择，持平保留较早一次；所有配置保存实际路径、参数量与初始化对齐方法。测试留到第五周。

数据读取由 `src/petvision/data/datasets.py::build_cifar10_loaders` 返回 45k/5k/10k 的 train / val / test loader；`scripts/inspect_cifar10.py::main` 创建两个 transform 后传入该函数，打印子集数量并检查批次类型和形状。具体签名与固定划分步骤见[Day 1 数据步骤](../../../docs/week3.md#cifar-data)。

配置中的 validation_size 和 num_workers 传给 build_cifar10_loaders，归一化 / 增强参数用于创建 `torchvision.transforms.v2` 变换；正式划分固定为 45000/5000。训练采用 ToImage → RandomCrop → RandomHorizontalFlip → ToDtype(torch.float32,scale=True) → Normalize；验证和测试去掉随机裁剪与翻转；`scripts/train_cv.py::main` 和 `scripts/evaluate.py::main` 的 resnet18 分支在 ToDtype 之前增加 Resize(96,96)，CNN 分支保留 32×32。配置中的 data.image_size 记录原图尺寸 32。完整示例与数值范围见 [v2 预处理步骤](../../../docs/week3.md#cifar-v2-transforms)。逐轮追加 history、耗时与峰值显存保存到 run_summary.json，具体见[资源记录步骤](../../../docs/week3.md#training-resources)。

初始状态按[训练前保存步骤](../../../docs/week3.md#initial-state)产生：Day 2 写 `outputs/cnn_bn_on/initial.pt`，Day 4 写 `outputs/resnet_on/initial.pt`，均在正式训练的第一次前向前保存，step=0。smoke/overfit 的模型不用于这些文件。

Day 5 在 BN off 配置设置 `training.init_checkpoint: outputs/cnn_bn_on/initial.pt`，在 residual off 配置设置 `training.init_checkpoint: outputs/resnet_on/initial.pt`。这两个字段在 `scripts/train_cv.py::main` 的模型构造之后、optimizer 创建之前读取：先检查来源 step=0，逐 key 核对 shape 并复制；残差对照要求模型状态键完全相同，BN 对照只允许预期的 BN 键差异。复制后由 run_training 保存各自 initial.pt，再从新优化器、零进度开始训练；best/last 用于训练结果和评估。文件不存在时先补齐该对基线，不能替换成训练后权重。运行命令不变，这个字段由配置读取。数据顺序使用独立 Generator；相同 seed 只能作为随机设置记录，若未独立控制随机增强，不声称逐样本增强完全一致。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| [cifar10_split.json](cifar10_split.json) | Day 1：训练/验证固定索引 |
| [configs/](configs/) | Day 5：四组实际配置快照 |
| [results.csv](results.csv) | Day 6 第一步：四组验证结果、预算与来源路径 |
| [figures/bn_comparison.png](figures/bn_comparison.png) | Day 6 第二步：BN 对照曲线 |
| [figures/residual_comparison.png](figures/residual_comparison.png) | Day 6 第二步：残差对照曲线 |
| `runs/<variant>/` | 各组配置快照、逐轮追加的 history、run_summary.json 与单组图 |
| 各组 `figures/mistakes_val.png` | Day 6：验证误分类，位于各自 experiment_dir |
| `outputs/{cnn_bn_on,cnn_bn_off,resnet_on,resnet_off}/` | 项目根下各组 initial（step=0）和 best/last checkpoint |

各组实际配置、运行材料和 best 链接见[第五周交接清单](#week5-handoff)，本周汇总表、对照曲线与验证分析见上方 Day 6 记录。大权重与原始数据保存在忽略目录。

<a id="weekly-summary"></a>

## 周总结（2026-11-04）

完成情况。 已统一 MNIST / CIFAR-10 的加载接口，将训练与评估变换分开；实现三层 CNN、按 D2L 结构组装的 ResNet18、BN 开关与残差开关。四组均完成 100 轮训练，每对共有初始状态已核对，四份 best 均经 CUDA、FP32、完整 5000 条验证样本独立复验。两张对照曲线、四组验证错图、[结果汇总表](results.csv)和[第五周交接材料](#week5-handoff)齐备。已有测试验收记录为 46 passed；[重写草稿](../../../tmp/basic_block_rewrite.py)的普通块与下采样块均通过实际前向、反向及原实现对照。

关键实验结果。 以下均为同一固定验证集上的 best accuracy，best 按验证 accuracy 选取，持平保留较早一轮。

- BN 对照：cnn_bn_on 为 77.24%，cnn_bn_off 为 61.50%，相差 15.74 个百分点。结合[BN 曲线](figures/bn_comparison.png)，开启 BN 在本轮预算下拟合更快、验证准确率更高；on 组的验证曲线仍有波动，准确率更高与曲线更稳定要分别判断。
- 残差对照：resnet_on 为 91.02%，resnet_off 为 90.28%，相差 0.74 个百分点。on 组训练准确率首次达到 90% / 98% 分别在第 17 / 55 轮，off 为第 23 / 68 轮；[残差曲线](figures/residual_comparison.png)体现的差异主要是达到相同拟合程度所需的 epoch，本次最终验证差距较小。关闭残差后仍能取得较高准确率，不据此否定残差对优化的作用。

关键理解与易错点。

- use_1x1conv 负责将旁路的通道数和空间尺寸对齐，use_residual 负责是否计算旁路并相加。普通块显式 use_1x1conv=False 时仍保留恒等残差；主分支最后一个 BN 后先相加，再 ReLU。
- initial.pt 保存 step=0 的初始状态，供 on / off 对齐后从零进度训练；best.pt / last.pt 保存训练结果。用训练后的 best 代替 initial 会改变消融起点。
- data.image_size=32 表示原始 CIFAR 图片边长，实际输入为 CNN 的 32×32、ResNet 的 Resize 后 96×96。复验与后续推理计时都要沿用各组真实预处理。
- 8 张错图是展示数量，全验证集错误数分别为 1138 / 1925 / 449 / 486。样例中的高 softmax 得分也可能对应错误预测；这几张图不能代表全体类别混淆频率。

结论范围。 两对内部各自保持划分、共有初始状态、增强策略和训练预算一致；本轮只有 seed=42，随机增强决策未逐样本独立对齐。ResNet off 保留投影参数但跳过其计算，因此总参数量相同不代表执行的计算完全相同。跨 CNN / ResNet 的输入尺寸、参数量和 batch 不同，跨架构分数差不能直接解释为 BN 或残差带来的收益。当前结果用于描述本轮观察，测试集继续保留到第五周统一评估。

后续巩固。本周工程与实验材料已整理；独立解释和有限参考重写继续按下面几项自查：

1. 不看参考，写出普通块和下采样块的两条路径，说明每层的输出尺寸，并解释恒等旁路与投影旁路各自怎样传递梯度；同时检查草稿运行结果和自己的解释。
2. 结合 BN on 的验证波动和 ResNet 后期的 train / val 差距，巩固 BN 的 train / eval 行为、running statistics，以及训练拟合速度、验证准确率和过拟合的区别。
3. 按[第四周指南](../../../docs/week4.md)继续 Attention、Mask 和 LayerNorm；第五周复用本周固定划分、实际配置与 best 权重，统一测试前继续使用验证集。多种子复验可作为后续拓展，当前四组材料保持原有实验口径。
