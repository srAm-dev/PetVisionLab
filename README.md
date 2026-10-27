# PetVision-Lab

PetVision-Lab 是一个用于学习深度学习原理与实践的个人项目。通过 Python、NumPy 和 PyTorch，逐步理解模型结构、训练流程与评估方法，并以亲手实现和小型对照实验积累实践经验。

当前仓库已包含第一周基础练习与 NumPy 线性回归，以及第二周的 MNIST 固定划分、参数化 MLP、训练 / 验证函数、配置入口和日志绘图。MNIST 已完成含 checkpoint 的正式五轮训练，best 为第 5 轮；独立加载后在 5000 条验证样本上准确率为 96.42%，与训练日志一致。接下来按 Day6 进行独立测试。后续周次仍是待实现的学习计划，实验是否完成以实际代码、结果文件和周总结为准。

## 项目目标

- 理解模型中的核心计算，能够解释张量形状、梯度与信息流。
- 建立独立编写、调试、训练和评估小型模型的能力。
- 通过对照实验分析模型行为，保存可复现的结果与结论。

## 学习范围

| 方向 | 主要内容 |
| --- | --- |
| 编程与数学基础 | Python、NumPy、矩阵运算、导数、梯度下降与线性回归 |
| 训练流程 | PyTorch、自动求导、数据加载、训练与评估、checkpoint |
| 计算机视觉 | CNN、ResNet、ViT，以及归一化、残差和 Patch Size 对照 |
| Transformer | Self-Attention、多头注意力、Mask、LayerNorm、位置编码与 Encoder |
| 语言建模 | Bigram、Mini-GPT、BPE、采样策略与中断续训 |
| 预训练模型 | CLIP 图文对齐、LLM 推理、VLM 结构与视觉问答 |

完整 Encoder–Decoder、宠物数据集迁移学习、LLM / VLM 微调和 Gradio 应用作为后续拓展。具体任务安排与验收要求见[总学习计划](docs/StudyPlan.md)。

## 实践方法

采用“理解原理 → 手写代码 → 运行检查 → 对照实验 → 总结”的学习方式：

- 从小数据和小模型起步，将公式、代码和实际输出对应起来。
- 用形状检查、梯度检查和参考实现验证关键模块。
- 在清楚的假设与控制条件下开展实验，分析成功结果、错误与局限。
- 日常通过代码与结果自查，每周简要汇总结论和遗留问题。

## 仓库组织

| 路径 | 用途 |
| --- | --- |
| `docs/StudyPlan.md` | 总体学习路线、阶段任务与验收要求 |
| `docs/` | 逐日学习指南与技术说明 |
| `scripts/` | 练习实现与实验入口 |
| `src/petvision/` | 可复用 Python 包；已包含 MNIST 数据、MLP、分类训练 / 验证和随机种子工具 |
| `experiments/` | 按 `<主题>/weekN/` 保存每周说明与总结、环境快照、实验结果和小型图表 |
| `environment.yml` | 当前本地 Linux 环境导出快照，包含具体构建版本和 PyTorch / CUDA 依赖 |
| `pyproject.toml` | Python 包配置；当前仅定义构建方式、包发现和 Python 版本要求 |

模型与训练模块按学习任务逐步建立，长期目录设计见总学习计划。逐日指南包含跟做所需的环境准备、操作、命令与排错；每周实验目录的 README 保存实际环境、复现索引和结果。

首次准备环境请从[第一周环境步骤](docs/week1.md)开始；根目录环境快照含平台相关包与本机路径，跨平台安装需要按指南重新选择依赖。`pip install -e .` 只安装项目包，当前不会自动补齐 NumPy、PyTorch 等运行依赖。

## 文档导航

- [总学习计划](docs/StudyPlan.md)：了解学习范围、任务安排与实验设计。

逐日指南提供资料、安装步骤、练习操作、运行命令、自查与验收标准；每周 README 集中维护实际版本、复现命令索引、结果链接与周总结。

| 周次与主题 | 逐日指南 | 每周说明与总结 |
| --- | --- | --- |
| 第 1 周：Python、NumPy 与数学 | [第一周指南](docs/week1.md) | [第一周 README](experiments/basics/week1/README.md) |
| 第 2 周：PyTorch 与 MNIST | [第二周指南](docs/week2.md) | [第二周 README](experiments/mnist_mlp/week2/README.md) |
| 第 3 周：CNN、ResNet 与对照 | [第三周指南](docs/week3.md) | [第三周 README](experiments/cnn_vs_resnet/week3/README.md) |
| 第 4 周：Attention 与 Encoder | [第四周指南](docs/week4.md) | [第四周 README](experiments/transformer_encoder/week4/README.md) |
| 第 5 周：ViT 与视觉比较 | [第五周指南](docs/week5.md) | [第五周 README](experiments/vit_patch_size/week5/README.md) |
| 第 6 周：字符级 Mini-GPT | [第六周指南](docs/week6.md) | [第六周 README](experiments/mini_gpt/week6/README.md) |
| 第 7 周：BPE、采样与续训 | [第七周指南](docs/week7.md) | [第七周 README](experiments/mini_gpt/week7/README.md) |
| 第 8 周：CLIP、LLM / VLM 与总结 | [第八周指南](docs/week8.md) | [第八周 README](experiments/vlm_qa/week8/README.md) |
