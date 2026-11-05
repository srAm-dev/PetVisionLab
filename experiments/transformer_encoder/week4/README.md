# 第四周：手写 Attention 与 Transformer Encoder

学习指南：[第四周逐日任务](../../../docs/week4.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本周尚未记录实测结果。先按第四周指南实现 Attention 和 Encoder，再在这里保存参考误差、Mask 检查、张量流说明和未理解的部分。

<a id="environment"></a>

## 环境准备

沿用 `petvision` 和第二周的 torch / pytest，不新增大模型依赖。CPU float64 做小尺寸数值对照，GPU FP32 确认设备兼容；完整操作和 Mask 语义见[第四周指南](../../../docs/week4.md)。版本变化时记录 torch 版本和参考 API。

```bash
conda activate petvision
python -m pip check
```

<a id="commands"></a>

## 实现后的运行命令

以下是实现后的运行入口，在项目根目录、对应 Conda 环境中执行。首次学习按指南逐步创建脚本和配置，再运行相应命令。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/attention_basics.py` | 单头小矩阵演示 |
| Day 2-3 | `python -m pytest tests/test_attention.py -q` | 多头、Mask、参考输出/梯度 |
| Day 4 | `python -m pytest tests/test_layernorm.py tests/test_encoder.py -q` | LayerNorm 对照与 Block 检查 |
| Day 5 | `python -m pytest tests/test_encoder.py -q` | 位置编码、Block 与 Encoder |
| Day 6 | `python -m pytest tests/test_attention.py tests/test_layernorm.py tests/test_encoder.py -q` | 本周统一检查 |

## 计划配置与待保存结果

形状中 B 是批次序列数、T 是每条序列的位置数、C 是每个位置的特征数；heads 是头数，每头维度为 C/heads。Q/K 决定关注权重，V 提供被加权的信息；具体矩阵计算见[形状与术语说明](../../../docs/week4.md#attention-shapes)和[Day1 手算](../../../docs/week4.md#day-1)。

统一 `[B,T,C]` 输入，自写布尔 Mask 的 True=屏蔽；causal 与 padding 合并后逐 query 检查，任一行没有可关注 key 时明确抛出 ValueError。本轮使用保留首个有效 token 的右侧 padding；padding query 输出不要求为零。B=2、T=8、C=32、heads=4、FFN=128 为起始检查配置。参考运行关闭 dropout，使用相同权重、精度、输入与 Mask。记录每项最大误差、容差、是否使用 CPU/GPU；张量检查结果直接打印并在周末总结。

在 attention.py 的 MultiHeadSelfAttention.forward 中保留 softmax 后、dropout 前的权重供检查；FeedForward 使用 Linear→ReLU→Linear，内部不放 dropout。blocks.py 的 EncoderBlock 分别调用 addnorm1、addnorm2，由各自 AddNorm 在子层输出上做 dropout，再残差相加和 post-LN。位置编码在构造表之前校验 embed_dim 为正偶数。

数值对照写入 `tests/test_attention.py::test_attention_reference`，Mask 行为分别写入同文件的 test_attention_causal_mask、test_attention_padding_mask、test_attention_combined_masks 和 test_attention_rejects_fully_masked_query，输入与断言见[Day 2](../../../docs/week4.md#reference-check)和[Day 3](../../../docs/week4.md#day-3)。LayerNorm 参数在 normalization.py::LayerNorm.__init__ 注册，位置 pe 在 positional_encoding.py 的构造函数中注册为 buffer。ViT / GPT 复用 Attention、LayerNorm，在各自模型文件定义 ViTBlock / GPTBlock。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `tensor_flow.md` | Day 6：模型结构、维度、Mask 与子层说明 |
| `tests/test_attention.py` | 项目根下：Attention 数值与行为检查 |
| `tests/test_layernorm.py / tests/test_encoder.py` | 项目根下：归一化、位置编码、组合检查 |

运行后为结果补上链接、命令和配置；权重与原始数据留在忽略目录。

## 周总结

- 参考对照：待运行后填写输出、输入梯度和参数梯度的最大误差、容差及设备。
- Mask 与形状：待检查后填写未来隔离、padding、全屏蔽处理和张量流链接。
- 重写与问题：待填写自己重写的片段、检查结果和仍不理解的计算。

周末用约 10-20 分钟补充，引用已有文件；未完成项写明卡在哪一步。
