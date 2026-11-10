# 第四周：手写 Attention 与 Transformer Encoder

学习指南：[第四周逐日任务](../../../docs/week4.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

Day 1 的单头演示、Day 2 的多头 Attention 与官方参考对照、Day 3 的 Mask 接入和扰动测试，以及 Day 4 的 LayerNorm 检查已通过 CPU 检查，结果见下方记录。后续继续完成 Mask 官方参考对照并实现 Encoder；独立解释与重写仍需本人自查。

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

## Day 1 检查记录（2026-11-06）

本次使用 Conda 环境 `petvision`，解释器为 `/home/sun/miniforge3/envs/petvision/bin/python`，Python 3.12.15、PyTorch 2.14.1+cu130。检查进程中 CUDA 不可用，以下结果均来自 CPU float64，未检查 GPU。

从项目根目录运行上表的 `python scripts/attention_basics.py`，正常退出，脚本中的断言全部通过：

- 零 query：三个 key 的权重均为 1/3，输出为 `[[3,2]]`，与手算一致。
- 随机批次：Q/K/V 为 `[2,3,2]`，scores、weights 为 `[2,3,3]`，output 为 `[2,3,2]`；权重非负，每行和与 1 的绝对误差不超过 `1e-12`。
- `output.square().mean()` 得到 loss=0.09931340519599491。反向后 Q/K/V 梯度均存在、形状与输入一致且数值有限。

固定随机种子 42，缩放对照得到以下结果。“最大权重均值”指先取每个 query 行中的最大权重，再对本次六行求平均。

| head_dim | 未缩放的最大权重均值 | 缩放后的最大权重均值 |
| --- | --- | --- |
| 2 | 0.5454 | 0.4847 |
| 32 | 0.9215 | 0.6406 |
| 128 | 0.9338 | 0.5609 |

这组输入中，缩放后权重更平缓；它控制分数尺度，权重仍由 Q/K 决定。

本次还临时运行了以下补充检查，未新增到仓库的 pytest 文件：

- 非零 query：用普通 Python 的 exp 和加权求和独立计算，权重与输出的最大绝对误差均为 0。
- 二维、三维和四维输入：覆盖单条、batch=1/2 和额外前导维度；query 数为 2、key 数为 5，输出形状正确，逐个切片与批量计算一致。
- Q/K/V 数值梯度：CPU float64，形状分别为 `[1,2,3]`、`[1,3,3]`、`[1,3,2]`；`torch.autograd.gradcheck` 在 eps=1e-6、atol=1e-5、rtol=1e-3 下通过。
- 导入演示脚本时没有自动运行 main，入口保护正常。

代码与 Day 1 的运行要求一致，本次无需修改。继续学习前，自行复述 Q/K/V 的职责、softmax 的维度、`[2,3,2]→[2,3,3]→[2,3,2]` 的计算过程，以及为什么除以 sqrt(d)。

## Day 2 第二步修正（2026-11-06）

- 构造时检查 embed_dim、num_heads 是正整数且可以整除，非法输入明确抛出 ValueError。
- 投影层统一命名为 q_proj、k_proj、v_proj，前向调用同步修改，与指南和后续参考测试一致。
- forward 在计算前检查 Mask 是否为 None；Day 2 收到任何非 None Mask 都抛出 NotImplementedError，包含全 False 的单元素和多元素 Tensor。

在 `petvision` 环境从项目根目录执行 `python -m pytest tests/test_attention.py -q`，结果为 **14 passed**：一个形状与梯度测试覆盖 B=1/2、T=1/8 四组输入，九组非法构造参数被拒绝，四组 Mask 输入在执行投影前被拒绝。Day 1 演示脚本重新运行通过。

第二步检查时尚未新增官方数值对照测试，随后完成的对照见下方第三步记录。Day 2 的 Mask 拒绝测试已在 Day 3 替换，见下方 Mask 接入记录。

## Day 2 第三步：官方参考对照（2026-11-06）

在 `tests/test_attention.py::test_attention_reference` 中，以 `torch.nn.MultiheadAttention` 为参考，固定 C=32、heads=4、dropout=0、CPU float64、eval 模式。对 bias=False/True、B=1/2、T=1/8 共八组输入逐一检查。

Q/K/V 的 weight 和存在的 bias 按顺序拼接，复制到参考层的 in_proj 参数；out_proj 单独复制。两边使用数值相同的独立叶子输入，对输出与同一个固定 probe 点乘求和后分别反向。比较输出、每头权重、输入梯度及四个投影的全部参数梯度。

运行 `python -m pytest tests/test_attention.py -q -s`，结果为 **22 passed**，其中八项为新增参考对照。八组中各类比较的最大绝对误差如下，检查容差均为 `1e-6`：

| 比较项 | 最大绝对误差，约 |
| --- | --- |
| 输出 | 1.665335e-16 |
| 每头注意力权重 | 1.110223e-16 |
| 输入梯度 | 1.665335e-16 |
| 投影 weight 梯度 | 1.110223e-15 |
| 投影 bias 梯度 | 1.776357e-15 |

结果对应本轮固定种子和上述 CPU 设置；测试运行时会逐项打印形状、最大误差和容差。仅重跑参考对照可执行 `python -m pytest tests/test_attention.py::test_attention_reference -q -s`。本次尚未接入 Mask，也未验证 GPU 数值对照。

## Day 3 第一步：Mask 接入与手算核对（2026-11-10）

`MultiHeadSelfAttention.forward` 已在 scores 缩放之后、softmax 之前接入布尔 Mask，True 表示屏蔽：

- 检查 attn_mask 为 `[T,T]`、key_padding_mask 为 `[B,T]`，均为布尔 Tensor，且与 scores 位于同一设备。
- 将两种 Mask 按 OR 合并，扩展到 `[B,heads,T,T]`，用 masked_fill 将屏蔽位置设为负无穷。
- 合并后任一 query 的所有 key 都被屏蔽时，抛出 ValueError。

新增 `test_attention_causal_known_values`：B=1、T=4、C=8、heads=2，Q/K 为零，V 投影和输出投影使用单位矩阵，四个位置的输入特征分别全为 1、3、5、7。causal mask 下，两个头的权重都与指南中的逐行均分结果一致，输出特征依次为 1、2、3、4；检查使用 rtol=0、atol=1e-12。手算表和单独运行命令见[Day 3 第一步](../../../docs/week4.md#day-3)。

在 `petvision` 环境运行 `python -m pytest tests/test_attention.py -q -s`，结果为 **37 passed**。检查还覆盖 causal 的未来扰动、padding 扰动、两种 Mask 合并、全屏蔽输入拒绝、非法 Mask 校验，以及已有的无 Mask 输出与梯度参考对照。合并 Mask 的合法输入能够正常反向，输入与参数梯度均为有限值。

本轮为 CPU 检查，Python 3.12.15、PyTorch 2.14.1+cu130，CUDA 不可用。尚未增加带 Mask 的官方数值对照，也未运行 SDPA 语义转换或 GPU 检查。

## Day 3 第三步：扰动检查（2026-11-10）

完善 `test_attention_causal_mask`、`test_attention_padding_mask`、`test_attention_combined_masks` 和 `test_attention_rejects_fully_masked_query`。统一使用 B=2、T=4、C=8、heads=2、CPU float64、dropout=0、eval，随机例子固定种子 42。

causal 测试将输入位置 2、3 的特征加 10，比较前两个位置的输出；padding 测试只在第一条的位置 2、3 和第二条的位置 3 加 10，分别比较前 2、3 个有效 query。两次前向共用模型和 Mask。随后使用同样的输入去掉 Mask 重算，确认扰动能实际影响被比较的输出。

| 检查 | 有 Mask 的输出最大绝对差 | 无 Mask 的输出最大绝对差，约 |
| --- | --- | --- |
| causal，两条序列的前两个位置 | 0 | 2.072780 |
| padding，第一条的前两个位置 | 0 | 2.072780 |
| padding，第二条的前三个位置 | 0 | 1.824026 |

输出差的检查容差为 `1e-6`。causal 与 padding 的屏蔽位置权重均为 0，每行权重和在 `1e-12` 容差内为 1。padding query 的输出不要求相同或为零。

合并 Mask 后，权重形状为 `[2,2,4,4]`，屏蔽位置的最大权重为 0，行和与 1 的最大绝对差为 `2.220446e-16`；输出、输入梯度和所有参数梯度均有限。单独 attn_mask、单独 padding，以及两个合法 Mask 合并后出现全屏蔽 query 的三种输入，均按约定抛出 ValueError。

运行 `python -m pytest tests/test_attention.py -q -s`，结果为 **37 passed**，其中本步骤的四类检查共六项。运行时会打印上述误差、比较形状、容差和全屏蔽错误；单独运行方法见[Day 3 第三步](../../../docs/week4.md#day-3)。

## Day 4 第二步：LayerNorm 检查（2026-11-10）

完成 `tests/test_layernorm.py` 中的三个测试函数，使用 CPU float64：

- test_layernorm_known_values：输入 `[1,3]`，将 eps=1e-5 计入期望值，输出约为 `[-0.999995,0.999995]`，检查容差为 1e-12。另只修改一个 token 的一个特征，确认其他 token 的输出不变。
- test_layernorm_constant_input：参数化 C=1、2、32，输入 `[2,3,C]` 的常量 7；gamma=1、beta=0 时输出为零，输入与参数梯度均存在且有限。
- test_layernorm_reference：对 B=1/2、T=1/3、C=32 共四组输入，给自写层设置固定随机的 gamma/beta，复制到 nn.LayerNorm 的 weight/bias。两边使用同一输入数值、独立叶子 Tensor 和相同 probe，比较输出、输入梯度与两个参数的梯度。

首次运行在批量输入处报错，原因是 normalization.py 使用 math.sqrt 处理方差 Tensor。将其改为 torch.sqrt 后，批量计算与方差路径的梯度均能通过对照。

运行 `python -m pytest tests/test_layernorm.py -q -s`，结果为 **8 passed**。四组官方参考对照的最大绝对误差如下，容差为 1e-6：

| 比较项 | 最大绝对误差，约 |
| --- | --- |
| 输出 | 8.881784e-16 |
| 输入梯度 | 1.332268e-15 |
| gamma / weight 梯度 | 1.776357e-15 |
| beta / bias 梯度 | 0 |

本次检查只覆盖 LayerNorm；FFN、AddNorm 与 EncoderBlock 按第三、四步继续实现和测试。

## Day 4 第四步：FFN 与 EncoderBlock 检查（2026-11-10）

完成 `tests/test_encoder.py`，使用 CPU float64，固定随机种子 42：

- test_feedforward_token_independence：输入为 `[2,3,32]`，FFN 隐藏维度为 128。只改第一条序列的位置 1，确认出口形状不变、该位置输出发生变化，其他位置输出在 1e-12 容差内相同。
- test_encoderblock_forward_backward：B=1/2、T=3、C=32、heads=4、FFN=128、dropout=0。分别使用无 Mask、causal、padding、合并 Mask，共八组。检查两个 AddNorm 及其归一化参数独立，输出形状正确，输入和全部参数的梯度均存在且有限。反向使用随机 probe 与输出相乘后求和。
- test_encoderblock_masks：输入为 `[2,4,32]`，分别使用单独的 causal、padding。扰动位置 2、3，位置 0、1 的输出在 1e-12 容差内不变；去掉 Mask 后，比较位置的最大输出差均大于 1e-6。padding query 不参与不变性断言。
- test_encoderblock_combined_masks：两种 Mask 同时传入。两条序列分别将位置 1、2 设为 padding，位置 0、3 保持有效。检查 causal 时扰动位置 3、比较 query 0；检查 padding 时扰动被屏蔽的位置、比较 query 3。合并时比较位置输出不变；只移除被检查的 Mask 后，两条序列的输出差均大于 1e-6。
- test_encoderblock_train_eval：dropout=0.1，检查所有子模块的 training 标记随 train/eval 切换。train 下连续两次输出不同；eval/no_grad 下连续两次输出完全相同；最后切回 train，确认标记恢复。

首次运行发现 AddNorm.forward 缺少 return，导致 FFN 收到 None；补上返回值后通过检查。

复查后为 LayerNorm 增加最后一维校验，并补上四项非法输入检查，覆盖可能被广播的维度错误、普通维度不匹配和零维 Tensor。合并 Mask 的测试改为上述两个独立检查；临时模拟同时传入时漏掉 attn_mask 或 key_padding_mask，新测试均能发现错误。

运行 `python -m pytest tests/test_attention.py tests/test_layernorm.py tests/test_encoder.py -q`，结果为 **63 passed**：Attention 37 项、LayerNorm 12 项、FFN 与 Block 14 项。未运行 GPU 检查；完整 Encoder 的位置编码与堆叠留到 Day 5。

## 后续计划配置与待保存结果

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
