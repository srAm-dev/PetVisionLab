# 第四周：手写 Attention 与 Transformer Encoder

学习指南：[第四周逐日任务](../../../docs/week4.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 `petvision` 和第二周的 torch / pytest，不新增大模型依赖。CPU float64 做小尺寸数值对照，GPU FP32 确认设备兼容；完整操作和 Mask 语义见[第四周指南](../../../docs/week4.md)。版本变化时记录 torch 版本和参考 API。

```bash
conda activate petvision
python -m pip check
```

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/attention_basics.py` | 单头小矩阵演示 |
| Day 2–3 | `python -m pytest tests/test_attention.py -q` | 多头、Mask、参考输出/梯度 |
| Day 4 | `python -m pytest tests/test_layernorm.py tests/test_encoder.py -q` | LayerNorm 对照与 Block 检查 |
| Day 5 | `python -m pytest tests/test_encoder.py -q` | 位置编码、Block 与 Encoder |
| Day 6 | `python -m pytest tests/test_attention.py tests/test_layernorm.py tests/test_encoder.py -q` | 本周统一检查 |

## 配置与结果

统一 `[B,T,C]` 输入，自写布尔 Mask 的 True=屏蔽；causal 与 padding 合并后逐 query 检查，任一行没有可关注 key 时明确抛出 ValueError。本轮使用保留首个有效 token 的右侧 padding；padding query 输出不要求为零。B=2、T=8、C=32、heads=4、FFN=128 为起始检查配置。参考运行关闭 dropout，使用相同权重、精度、输入与 Mask。记录每项最大误差、容差、是否使用 CPU/GPU；张量检查结果直接打印并在周末总结。

Attention 返回供检查的 dropout 前权重；权重 dropout 由 Attention 管理，FFN 隐藏 dropout 由 FeedForward 管理，两条残差输出 dropout 由 EncoderBlock 管理，避免重复放置。位置编码在构造表之前校验 embed_dim 为正偶数。

构造 Attention 对象及权重复制见[Day 2](../../../docs/week4.md#reference-check)，Mask 创建与合并见[Day 3](../../../docs/week4.md#day-3)，可学习参数注册和位置 buffer 分别见 Day 4 / Day 5；ViT / GPT 复用 EncoderBlock 的类并创建新实例。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `tensor_flow.md` | Day 6：模型结构、维度、Mask 与子层说明 |
| `tests/test_attention.py` | 项目根下：Attention 数值与行为检查 |
| `tests/test_layernorm.py / tests/test_encoder.py` | 项目根下：归一化、位置编码、组合检查 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

