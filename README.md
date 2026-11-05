# PetVision-Lab

这是我用来学习深度学习的项目。从 Python、NumPy 和梯度下降开始，逐步手写 CNN、ResNet、Transformer、ViT 和 Mini-GPT，再体验预训练 LLM / VLM 推理。每个阶段都用小实验检查理解，保留运行结果和遇到的问题。

八周的目标是能独立实现、训练、调试和解释小模型，重点练习张量形状、梯度、训练评估和对照实验。完整 Encoder-Decoder、宠物数据集迁移、模型微调和 Gradio 应用放到后续学习。

第一次跟做，从[学习与执行入口](docs/README.md)进入；查看整体安排读[总计划](docs/StudyPlan.md)。已有代码时，按当前周的检查和重写练习复习，实际进度与结果看各周实验 README。

## 八周安排

| 周次 | 学习内容 | 逐日步骤 | 环境、命令与结果 |
| --- | --- | --- | --- |
| 1 | Python、NumPy、数学与线性回归 | [第一周](docs/week1.md) | [实验记录](experiments/basics/week1/README.md) |
| 2 | PyTorch、MNIST MLP 与训练流程 | [第二周](docs/week2.md) | [实验记录](experiments/mnist_mlp/week2/README.md) |
| 3 | CNN、ResNet、BN 与残差对照 | [第三周](docs/week3.md) | [实验记录](experiments/cnn_vs_resnet/week3/README.md) |
| 4 | Attention、Mask、LayerNorm 与 Encoder | [第四周](docs/week4.md) | [实验记录](experiments/transformer_encoder/week4/README.md) |
| 5 | ViT、Patch 对照与视觉模型比较 | [第五周](docs/week5.md) | [实验记录](experiments/vit_patch_size/week5/README.md) |
| 6 | 字符级 Bigram、Mini-GPT 与生成 | [第六周](docs/week6.md) | [实验记录](experiments/mini_gpt/week6/README.md) |
| 7 | 字节 BPE、采样与中断续训 | [第七周](docs/week7.md) | [实验记录](experiments/mini_gpt/week7/README.md) |
| 8 | CLIP、LLM/VLM 推理与总结 | [第八周](docs/week8.md) | [实验记录](experiments/vlm_qa/week8/README.md) |

每周安排 24 小时任务和 6 小时机动，按检查结果推进。训练结果、解释能力和核心代码重写都要练习；未通过的基础项优先补齐，后续任务顺延。

## 文件放在哪里

| 路径 | 用途 |
| --- | --- |
| `docs/` | 总计划、逐日步骤、接口约定与完成标准 |
| `scripts/` | 练习、训练、评估和推理入口 |
| `src/petvision/` | 可复用的数据、模型、训练和工具代码，随学习逐步增加 |
| `tests/` | 形状、梯度、Mask、保存加载等检查 |
| `experiments/<主题>/weekN/` | 实际进度、环境快照、配置、小型结果与周总结 |
| `outputs/`、`datasets/` | 大权重、日志和数据，按忽略规则保留在本地 |
| `environment.yml` | 当前本地 Linux 环境的完整导出快照 |
| `pyproject.toml` | 构建方式、包发现和 Python 版本要求 |

环境准备从[第一周](docs/week1.md#day-1)开始，第二周再补 PyTorch 和 GPU 依赖。现有 `environment.yml` 含平台构建与本机路径，跨平台按指南选择依赖。`pip install -e .` 只安装本地项目包，运行依赖需要另行安装。
