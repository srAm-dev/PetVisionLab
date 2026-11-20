# 学习与执行入口

第一次学，从第一周 Day 1 开始。先用 Python 算三条样本的均方误差，再用 NumPy 做一次矩阵预测。每周安排 24 小时学习和 6 小时机动。遇到算不通的例子，记下卡住的那一步，下次从那里继续。[第一天练习](week1.md#day-1) · [八周总计划](StudyPlan.md#overview)

<a id="resume"></a>

## 从仓库现有进度接着学

**第四周 Day 6 已完成，下一次从第五周 Day 1 开始。** 本周 69 项测试通过，三种 Mask 的数值对照和多头 Attention 重写都已有运行结果。[第四周实验记录](../experiments/transformer_encoder/week4/README.md#day-6-results) · [两层 Encoder 张量流](../experiments/transformer_encoder/week4/tensor_flow.md)

第五周先手算一张 32×32 的 RGB 图能切出多少个 4×4 和 8×8 的 Patch，再实现 `PatchEmbedding`，分别核对 64 和 16 个图像 token。加入分类 token 后，序列长度变为 65 和 17。[第五周 Day 1](week5.md#day-1)

## 每周从哪里接手

| 逐日指南 | 从哪里开始 | 学完用什么检查 | 实验记录 |
| --- | --- | --- | --- |
| [1　NumPy 回归](week1.md) | 在 CPU 上算 `X @ w + b`，写出预测和标签的形状 | 用中心差分核对两个权重和偏置的梯度，再用同一份数据比较三个学习率 | [回归曲线与结果](../experiments/basics/week1/README.md) |
| [2　MNIST 分类](week2.md) | 把第一周的 MSE 梯度与 `torch_basics.py` 的 autograd 结果逐项比较 | 先拟合固定 64 个样本，再训练 MLP。用验证准确率选 best，冻结后在 10,000 张测试图片上评一次 | [MNIST 记录](../experiments/mnist_mlp/week2/README.md) |
| [3　CNN 与 ResNet](week3.md) | 沿用第二周的分类训练循环，为 CIFAR-10 固定 45,000/5,000 的训练与验证划分 | 用同一批 5,000 张验证图，分别比较 CNN 打开或关闭 BN、ResNet 打开或关闭残差的结果。为四个 best 留下各自的训练配置 | [四组模型交接表](../experiments/cnn_vs_resnet/week3/README.md#week5-handoff) |
| [4　Attention 与 Encoder](week4.md) | 从 BasicBlock 的残差相加过渡到序列，先手算零 query 对三个 value 的加权平均 | 对照官方层的输出与梯度，扰动后缀检查 causal Mask，沿两层 Encoder 解释张量形状 | [Attention 与 Encoder 记录](../experiments/transformer_encoder/week4/README.md) |
| [5　ViT 与视觉比较](week5.md) | 用第四周的 Attention 和 LayerNorm 组 ViTBlock，训练时沿用第三周的划分 | 把 32×32 图片切成边长为 4 或 8 的 Patch，加上分类 token 后，序列长度分别是 65 和 17。训练这两组 ViT，与第三周四个 best 一起冻结后，在同一测试集上比较六组模型 | [视觉实验记录](../experiments/vit_patch_size/week5/README.md) |
| [6　字符语言模型](week6.md) | 沿用第四周的因果 Mask 和第二周的保存加载工具。先切分原文，再建字符词表和错位窗口 | 同词表的 Bigram 与 Mini-GPT 在相同验证目标上比较 loss，重新加载各自的 best 后从固定 prompt 生成 | [字符模型记录](../experiments/mini_gpt/week6/README.md) |
| [7　BPE、采样与续训](week7.md) | 取第六周的原文划分、字符词表和两个 best。用训练文本学 BPE，用字符 GPT 做采样对照 | 检查 Unicode 编解码往返和 CPU 小模型从第 21 步续训。冻结三个模型后测试，BPE 的 token loss 单列 | [BPE 与续训记录](../experiments/mini_gpt/week7/README.md) |
| [8　预训练推理](week8.md) | CLIP 用第三周验证索引取原图，LLM/VLM 用各自配套的 tokenizer 或 processor。第五至七周的结果用于写报告 | CLIP 完成 200 图零样本分类。VLM 先跑通单图，再按预先写好的标准判定至少 20 图、每图至少两题的全部回答 | [推理记录与报告入口](../experiments/vlm_qa/week8/README.md) |

<a id="working-rules"></a>

## 跟做时的约定

所有命令都从项目根目录执行，先激活对应 Conda 环境。指南中的代码路径也从根目录算起。

| 位置 | 放什么 |
| --- | --- |
| `scripts/` | 可运行的练习、训练和推理入口 |
| `src/petvision/` | 可复用的数据、模型、训练和工具函数；新增包目录补 `__init__.py` |
| `tests/` | 检查形状、梯度、Mask、保存加载等行为的测试 |
| `experiments/<主题>/weekN/` | 配置快照、小型指标、图表、环境记录和周总结 |
| `outputs/`、`datasets/` | 权重、大日志和数据，按 `.gitignore` 保留在本地 |

遇到已有实现，先预测一个小输入的输出，再读源码、运行当天的检查。重写练习可用纸笔或 `tmp/`。复查训练结果时用当次的配置快照和权重。确需重训时，先复制配置，为 `experiment_dir` 和 `output_dir` 指定新的目录。

第五至八周的实验 README 已备好。按日程先实现被调用的模块，再接配置和脚本，最后执行相应命令。代码里的 `...` 和 `pass` 是需要填入计算的位置。自第四周 Day 5 起，每项新计算用一个代表输入检查，完整模型做一次前向和反向。形状字母和各阶段的训练要求见[总计划约定](StudyPlan.md#reading-conventions)。

## 怎样判断今天做完了

先核对当天“运行与自查”中的输出、断言和样本数，再用自己的话解释计算过程。脚本能跑通与能自己重写分别检查。训练前先做最小前反向和小样本拟合，确认后再扩大预算。

结果写入对应周 README，并链接当次配置、指标和图表。日常可以用口头说明、纸上推导或代码注释检查理解；周末用约 10-20 分钟补充结果、一个遇到的问题和下周要巩固的内容。环境变化时再更新快照。

遇到报错，先停在能复现问题的最小输入，检查形状、标签、设备、梯度和运行模式。比较公式或参考实现时固定输入、权重、Mask 和精度。准确率未达参考值时，先检查流程，再如实分析预算与错误样例；实验结论可以不支持最初的假设。

## 实验之间怎样衔接

做成对比较时，固定划分、初始化方式、增强策略、优化器和训练预算。跨模型比较还要写明各自的输入尺寸，本项目的 CNN 和 ViT 用 32×32，教材 ResNet 用 96×96。文本实验先切分原文，再造窗口，BPE 只从训练原文学规则。字符和 BPE 的一个 token 表示的文本长度不同，两者的 token loss 分开报告。[实验设计](StudyPlan.md#acceptance)

总计划和逐日指南保存任务、接口及完成标准，实际进度只在实验 README 中维护。第八周在 `docs/final_report.md` 汇总证据和待补项。完整 Encoder-Decoder、宠物迁移学习、LoRA/QLoRA、VLM 微调和 App 见[后续拓展](StudyPlan.md#extensions)，另行排期。
