# 第四周逐日学习指南：手写 Attention 与 Transformer Encoder

> 更新日期：2026-10-24  
> 对应计划：[总学习计划](StudyPlan.md#stage-2)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：实现可复用多头 Attention、Mask、LayerNorm、FFN、位置编码和 Encoder，通过形状、归一化、梯度、未来隔离和官方参考对照。  
> 环境：沿用 Conda `petvision`；先在 CPU float64 小张量上做数值对照，再做本地 GPU FP32 检查。本周不要求大规模训练。

## 怎么使用这份指南

先确认[上一周指南](week3.md)的基础关卡已通过。应能解释 nn.Module、反向传播与残差形状。本周的模块将在第五周 ViT 和第六周 Mini-GPT 中复用。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/transformer_encoder/week4/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── src/petvision/models/transformer/
│   ├── __init__.py
│   ├── attention.py                      # Day 1–3：单头、多头、Mask
│   ├── normalization.py                  # Day 4：简化 LayerNorm
│   ├── positional_encoding.py            # Day 5：正弦位置编码
│   └── blocks.py                         # Day 4–5：FFN、EncoderBlock、Encoder
├── scripts/attention_basics.py            # Day 1：小矩阵手算对照
├── tests/
│   ├── test_attention.py                 # Day 2–3：Mask 与参考 Attention
│   ├── test_layernorm.py                 # Day 4：输出与梯度
│   └── test_encoder.py                   # Day 4–5：Block、组合模块与模式切换
└── experiments/transformer_encoder/week4/
    ├── README.md                         # 已提供：命令与周总结
    └── tensor_flow.md                    # Day 6：模块结构与张量形状
```

## 开始前：激活环境并检查已有模块

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

解释器应位于 Conda 的 `petvision` 环境，`petvision.__file__` 应指向当前仓库的 `src/petvision/`。若出现 `No module named petvision`，先确认第二周已建立 `pyproject.toml` 和包目录，再在根目录执行 `python -m pip install -e . --no-deps`；若缺少 torch 或 CUDA 不可用，按[第二周 Day 1 的安装与排错步骤](week2.md#environment)逐项核对。

先用 CPU float64 小张量检查数值，再确认 GPU FP32 可运行。 本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

教学模块统一输入 `[B,T,C]`，起始 B=2、T=8、C=32、heads=4、head_dim=8、FFN hidden=128；要求 C 能被 heads 整除。参考检查 dropout=0、相同 dtype、相同权重和输入。Encoder 使用 pre-LN：`x=x+Attention(LN(x))`，再 `x=x+FFN(LN(x))`；与原论文 post-LN 的位置区别要能说明。参考 `nn.TransformerEncoderLayer` 时设置 batch_first=True、norm_first=True、activation='gelu'，对齐 eps、bias 和 FFN 维度。

自写布尔 Mask 统一约定 True=屏蔽，在 softmax 前填 -inf；causal mask 为 `[T,T]` 严格上三角，padding mask 为 `[B,T]`，广播到 key 维。`nn.MultiheadAttention` 布尔 Mask 的 True 也是屏蔽；`F.scaled_dot_product_attention` 则是 True=允许关注，对照时需要转换。本轮接口只接收上述布尔 Mask，浮点加性 Mask 不在教学接口范围。合并后每个 query 都必须有至少一个可关注 key（包括 padding query），否则明确拒绝；padding query 的输出不要求自动为零。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | 缩放点积 Attention 与 softmax | 4h | 1h | 单头注意力手算和数值对照 |
| Day 2 | 多头拆分与官方 Attention 对照 | 4h | 1h | 手写 QKV 投影、多头合并 |
| Day 3 | Padding / Causal Mask 与未来隔离 | 4h | 1h | 屏蔽规则与扰动测试 |
| Day 4 | LayerNorm、FFN 与 pre-LN Block | 4h | 1h | 简化 LayerNorm 与 EncoderBlock |
| Day 5 | 位置编码与 Encoder 堆叠 | 4h | 1h | 完整 Encoder、组合检查 |
| Day 6 | 统一检查、张量流与有限参考重写 | 4h | 1h | 核心模块与形状图可复用 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：缩放点积 Attention 与 softmax

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 3.2.1 与 Attention 公式 | 对应 Q、K、V 与缩放项 |
| [D2L 注意力评分函数](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html) | 缩放点积注意力的中文解释；本日先不展开其 Mask 实现 | 把公式与小矩阵对应 |
| [PyTorch softmax（中文译文）](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.functional.softmax.html) | dim 维度含义 | 检查每行权重和 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读 Q/K/V 与 softmax，手算小矩阵 |
| 2h | 实现缩放点积核心与演示 |
| 1h | 核对形状、归一化与梯度 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `scripts/attention_basics.py`，取短序列 T=3、head_dim=2 的 Q、K、V，先手算 `scores=Q@K.T/sqrt(head_dim)`，沿最后的 key 维 softmax，再乘 V。

2. 在 `attention.py` 实现基础函数，支持批次维；解释 `[B,T,d]→[B,T,T]→[B,T,d]`。用 torch.softmax 保证稳定性，理解它在内部处理大 logits 的数值问题。

3. 比较缩放前后 scores 的范围与权重集中程度，解释除以 sqrt(d_k) 的作用；这一步为观察，不扩成超参数实验。

4. 打印输出、注意力权重及其行和，反向后检查 Q/K/V 梯度有限。演示放在入口保护下。

### 分步操作与自查

#### 第一步：用一个能手算的例子理解加权平均

创建 `src/petvision/models/transformer/`，放入空的 `__init__.py`，再创建 `attention.py`；同时创建 `scripts/attention_basics.py`。可复用计算写在 `src/petvision/models/transformer/attention.py`，手算例子、打印和断言写在脚本的 main 中，末尾用入口保护调用；从 `petvision.models.transformer.attention` 导入自己写的函数。

取一个 query，令 Q 全零、K 任意，V 三行为 `[1,0]`、`[3,2]`、`[5,4]`。scores 全为 0，softmax 后每个 key 权重为 1/3，输出应为 `[3,2]`。

```text
Q                [1,2]
K / V            [3,2]
Q @ K.T          [1,3]
weights @ V      [1,2]
```

先用这个例子检查 softmax 是否沿最后一维、是否把 V 误写成 K。加 batch 维后，计算含义不变。

#### 第二步：分开保存中间量

在 `src/petvision/models/transformer/attention.py` 定义 `scaled_dot_product_attention(q, k, v)`，将下方公式整理为函数并返回 `(output, weights)`；函数内使用参数 q/k/v，脚本中的大写 Q/K/V 是传给它的示例张量。Day 1 先不加 Mask 与 dropout；Day 2–3 再完善多头模块。

```python
import math
import torch

scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
weights = torch.softmax(scores, dim=-1)
output = weights @ V
```

打印 scores.shape、weights.shape、output.shape。softmax 输出每行非负、和约为 1；不要沿 query 维归一化。

#### 第三步：理解缩放而非只背公式

观察 head_dim 变大时点积可能出现更大的数值，较大的 scores 差异会使 softmax 更集中。除以 sqrt(head_dim) 用于控制点积的尺度。

先以小随机张量对输出求标量损失并反向，确认 Q/K/V 梯度存在且有限；正式模型中的 Q/K/V 将来自可学习投影。

随机检查时直接创建 float64、`requires_grad=True` 的 Q/K/V 叶子 Tensor，形状均为 `[2,3,2]`。先断言 weights 的最后一维和为 1，再用 `output.square().mean()` 作标量目标，backward 后检查三者的 grad 非 None、同形状且有限。零 query 的例子用于核对数值；随机例子用于检查一般计算路径，零梯度本身不算错误。

### 运行与验收

```bash
python scripts/attention_basics.py
```

先核对零 query 的例子：三个 key 的权重均约为 `1/3`，输出为 `[3,2]`。再运行随机 Q/K/V 例子，打印 scores、weights、output 的形状和权重行和。若输出尺寸正确但数值不对，逐步核对转置、softmax 的维度及最后乘的是 V。

- [ ] 能说明 Q 与 K 决定权重，V 提供被加权的信息。
- [ ] softmax 沿 key 维，未加 dropout 的权重和约为 1。
- [ ] 小矩阵计算和 Tensor 形状能逐项解释。

<a id="day-2"></a>
## Day 2：多头拆分与官方 Attention 对照

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 3.2.2 Multi-Head Attention | 画出拆分与拼接 |
| [D2L 多头注意力](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html) | 多头投影、拆头与拼接的形状 | 先列出下面例子的每一步尺寸 |
| [nn.MultiheadAttention（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.MultiheadAttention.html) | batch_first、权重形状与返回注意力 | 建立只作对照的参考层 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读多头维度与参考接口 |
| 2h | 实现投影、reshape、transpose 和合并 |
| 1h | 复制权重，比较输出与梯度 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 完成 `MultiHeadSelfAttention`：分别 Linear 得到 Q/K/V，变为 `[B,heads,T,head_dim]`，每头计算注意力，合并回 `[B,T,C]` 后做输出投影。

2. 实现可选 `return_weights`，返回 dropout 前的 softmax 权重供检查；模型本体不调用 nn.MultiheadAttention 或 SDPA 来替代手写核心。

3. 创建 `tests/test_attention.py`，覆盖 batch=1/2、不同 T 与 C%heads!=0 的明确报错；输出形状与每个参数的梯度正确。

4. 参考层用 batch_first=True、dropout=0。把自写 QKV 按正确顺序拼到 in_proj_weight/bias，对齐 out_proj；使用相同标量 loss 比较输出、输入梯度和映射后的参数梯度。float64 起始容差 1e-6，记录实际最大误差。

### 分步操作与自查

#### 第一步：把 reshape 与 transpose 分成两步

例如 B=2、T=3、C=8、heads=2，每头维度为 4：

```text
Linear 后 Q           [2,3,8]
reshape               [2,3,2,4]
transpose(1,2)        [2,2,3,4]
Q @ K.transpose(-2,-1)[2,2,3,3]
乘 V 后               [2,2,3,4]
合并 heads            [2,3,8]
```

transpose 后内存可能不连续，合并时用 contiguous().view 或 reshape，不能只改形状而漏掉头/序列轴的交换。

#### 第二步：明确模块接口

继续修改 `src/petvision/models/transformer/attention.py`，定义继承 nn.Module 的 MultiHeadSelfAttention；投影层在 __init__ 中创建，下方是类的方法，不能作为文件顶层的普通函数。

```python
def forward(self, x, attn_mask=None, key_padding_mask=None,
            return_weights=False):
    # x: [B,T,C]；返回输出及可选的 dropout 前权重
    ...
```

类的构造接口约定为 `MultiHeadSelfAttention(embed_dim, num_heads, dropout=0.0, bias=True)`；embed_dim 对应 C，num_heads 对应 heads。默认 forward 只返回输出 Tensor；return_weights=True 时返回 `(output, weights)`，weights 为 `[B,heads,T,T]` 的 dropout 前权重，EncoderBlock 使用默认返回值。

参数包含 q_proj、k_proj、v_proj、out_proj。C%heads 不为 0 时在构造阶段报错，避免后续 reshape 才出现模糊异常。

<a id="reference-check"></a>
#### 第三步：逐项复制到官方参考

创建 `tests/test_attention.py`，顶层导入 pytest、torch 和自己模块中的 MultiHeadSelfAttention。先写 B=1/2、不同 T、非法头数和梯度检查，再把下面的官方参考对照放进独立 `test_` 函数。

先在检查函数中创建两个独立对象；MultiHeadSelfAttention 从自己实现的 attention.py 导入：

```python
attention = MultiHeadSelfAttention(embed_dim=32, num_heads=4, dropout=0.0).double()
reference = torch.nn.MultiheadAttention(
    embed_dim=32, num_heads=4, dropout=0.0, batch_first=True,
).double()
```

参考 in_proj 的前三段依次为 Q、K、V；weight 按 dim=0 拼接，bias 同顺序。out_proj 的 weight/bias 单独对应。

例如自写模块名为 `attention`，参考层名为 `reference` 时，权重映射的核心关系为：

```python
with torch.no_grad():
    reference.in_proj_weight.copy_(torch.cat([
        attention.q_proj.weight,
        attention.k_proj.weight,
        attention.v_proj.weight,
    ], dim=0))
    reference.out_proj.weight.copy_(attention.out_proj.weight)
```

bias 也按 Q/K/V 同顺序复制，输出 bias 单独复制。两边先都转为 CPU float64；输入分别使用数值相同、各自 `requires_grad=True` 的叶子 Tensor，避免把两次反向的梯度累加到同一个输入上。

关闭 dropout，用同一输入分别前向；对两个输出使用相同标量目标再 backward，比较输入梯度和对应参数梯度。仅输出相近不能代替完整梯度检查。

把这句话落实为一个检查函数，顺序固定如下：

1. 两个模块都转为 CPU float64，dropout=0，复制全部 Q/K/V 和 out_proj 的 weight/bias，清除两边已有梯度。
2. 先生成一份数值 x，再分别用 `x.detach().clone().requires_grad_(True)` 创建 `x_custom、x_reference`。参考层调用 `reference(x_reference, x_reference, x_reference)`，取其返回的第一个元素作为输出。
3. 生成一份与输出同形状的固定 probe；两边都用 `(output * probe).sum()` 作为目标，各自 backward。probe 数值相同，不能各自重新抽样。
4. 比较输出、两个输入的 grad、Q/K/V 和 out_proj 的参数梯度；每项用 `(actual-expected).abs().max().item()` 求最大绝对差，打印并断言小于 `1e-6`。

参考 `in_proj_weight.grad` 的 `[:C]、[C:2*C]、[2*C:]` 分别对应自写 q/k/v 的 weight.grad；bias.grad 同样分段，out_proj 直接对应。每项先检查 shape、梯度非 None 且有限，再做数值比较。后面的 LayerNorm 与 Encoder 对照沿用这套「相同权重、独立输入、同一目标」流程。

#### 第四步：打印真正有用的结果

输出每项最大绝对误差、容差和 shape，不打印成千上万个权重。注意力行和使用 softmax 后、dropout 前的权重。

### 运行与验收

```bash
python -m pytest tests/test_attention.py -q
```

今天检查多头形状、非法 heads、输出和梯度对照；Mask 检查在明天添加。可给对照检查加 `-s` 查看自行打印的误差，通常使用 `-q` 查看简洁结果。float64 起始绝对容差为 `1e-6`，应同时报告实际误差，不能只写“通过”。

- [ ] 能解释 reshape 与 transpose 各改变了什么。
- [ ] 权重映射完整，参考输出和梯度均有数值证据。
- [ ] 关闭 dropout 时注意力归一化检查通过。

<a id="day-3"></a>
## Day 3：Padding / Causal Mask 与未来隔离

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [nn.MultiheadAttention Mask（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.MultiheadAttention.html) | attn_mask、key_padding_mask 的形状与布尔语义 | 对齐自写 Mask |
| [SDPA 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 布尔 Mask 与 dropout_p 语义 | 识别参考接口差异 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 画出上三角和 padding key 屏蔽图 |
| 2h | 实现合并 Mask 与扰动检查 |
| 1h | 核对官方参考和边界行为 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 在 softmax 前应用 causal mask，禁止位置 i 关注 j>i；padding mask 屏蔽无效 key 列。明确广播到 `[B,heads,T,T]` 的步骤。

2. 先合并并广播两种 Mask，逐 query 检查是否存在可关注 key；仅保证整条序列有有效 key 仍不够。对于全屏蔽查询明确抛出 ValueError，避免全 -inf 行产生 NaN。检查有效查询的屏蔽权重接近 0、可用 key 权重和约 1。

3. 在 eval、dropout=0 下复制输入，只改变位置 k 之后的内容；开启 causal 时，输出前 k+1 个位置应保持一致。关闭 causal 后不要求该性质。

4. 单独改变 padding key 的输入，验证有效 query 输出保持一致；padding query 本身的输出可非零。与 nn.MultiheadAttention 对照时传一致的显式 Mask；若对照 SDPA，转换布尔语义并显式 dropout_p=0。

### 分步操作与自查

#### 第一步：在纸上画 causal mask

今天继续修改 `src/petvision/models/transformer/attention.py` 的 forward，将 Mask 处理放在 softmax 之前；以下扰动、权重与异常检查添加到已有 `tests/test_attention.py`，保留 Day 2 的参考对照。

本项目 True 表示屏蔽，T=4 时为：

```text
query\key  0 1 2 3
0          F T T T
1          F F T T
2          F F F T
3          F F F F
```

第 i 个位置可以看自己与过去。上三角应从 diagonal=1 开始；若把对角线也屏蔽，第一个位置将没有可关注 key。

#### 第二步：区分 key padding 与 query 位置

padding mask `[B,T]` 扩展到 `[B,1,1,T]`，作用在 key 列。先给接口传入真实布尔 Tensor，例如 T=4 的 causal mask 用 `torch.ones((4,4), dtype=torch.bool, device=x.device).triu(diagonal=1)` 创建；两条序列的 padding mask 可用 `[[False,False,True,True],[False,False,False,True]]` 创建，并放在相同设备。

在 attention.forward 计算出 scores 后，合并步骤为：先建立与 scores 同设备、形状 `[B,1,T,T]` 的全 False 布尔 Tensor；若有 attn_mask，OR 上 `attn_mask[None,None,:,:]`；若有 key_padding_mask，OR 上 `key_padding_mask[:,None,None,:]`；最后 `blocked_mask = blocked_mask.expand_as(scores)`。两种 Mask 都未提供时不屏蔽任何位置。先检查传入的 shape、dtype、device，再执行下方填充，不能直接使用尚未构造的 blocked_mask。

```python
# blocked_mask 是已合并并广播到 scores.shape 的布尔 Tensor。
if blocked_mask.all(dim=-1).any().item():
    raise ValueError("合并 Mask 后存在没有可关注 key 的 query")
scores = scores.masked_fill(blocked_mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
```

有效 query 的屏蔽位置权重应为 0、其余权重和约为 1。本轮示例使用至少保留首个有效 token 的右侧 padding；左侧 padding 与 causal 合并后，最前面的 padding query 可能没有可关注 key，即使序列后面有有效 token 也要按上述约定拒绝。支持这类输入时需另行定义无效 query 的处理方式。

#### 第三步：通过扰动验证因果性

复制 x，仅改第 k+1 个及之后 token；两次使用同一个模型、eval、dropout=0。比较输出 `[:, :k+1]`，应在容差内一致。

padding 检查同理，只改被屏蔽 key 的值，比较有效 query 输出。不要拿 padding query 的输出必须为 0 当作一般规则。

先使用固定小例子 B=2、T=4、C=8、heads=2：

| 检查 | 怎样修改输入 | 比较哪些输出 |
| --- | --- | --- |
| causal | clone x，将位置 2、3 加上固定扰动；k=1 | 两次输出的 `[:, :2]`，最大绝对差小于 `1e-6` |
| padding | 第一条屏蔽 key 2、3，第二条只屏蔽 key 3；只改这些位置 | 分别比较第一条前 2 个、第二条前 3 个有效 query |
| 合并 | 将上述 padding 与 causal 按 OR 合并 | 权重在合并屏蔽处为 0，其余行和为 1，输出有限 |
| 全屏蔽 | 构造某个 query 没有有效 key 的输入 | 用 `pytest.raises(ValueError)` 检查约定的明确拒绝 |

每个检查使用同一模型、eval、dropout=0 和两份 clone 输入。需要比较权重时通过 return_weights 取 dropout 前权重，先将 Mask `expand_as(weights)`，再检查被屏蔽位置和每行和；合法例子确保每一行至少有一个有效 key。上表检查的是 Attention 本身，后面堆叠模块还要验证 Mask 传递。

#### 第四步：和参考对照时转换语义

nn.MultiheadAttention 的布尔 True 为屏蔽；SDPA 的 True 为允许关注。测试前显式转换，而不是把同一个布尔 Tensor 原样传给两种语义相反的接口。SDPA 的 `dropout_p` 由调用者显式传入，评估时设置为 0；模型切 eval 不会自动覆盖这个函数参数。具体语义见 [PyTorch 官方 SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

### 运行与验收

```bash
python -m pytest tests/test_attention.py -q
```

今天补上 causal、padding、两种 Mask 合并与全屏蔽输入检查。若未来扰动后前缀输出改变，先确认 dropout 已关闭，再查上三角方向和 Mask 是否作用在 softmax 前；若出现 NaN，检查是否存在整行 key 被屏蔽。

- [ ] 屏蔽发生在 softmax 前，Mask 形状与布尔含义有明确约定。
- [ ] 未来隔离与 padding key 扰动测试通过。
- [ ] 能解释 Encoder 双向 Attention 与 GPT causal Attention 的区别。

<a id="day-4"></a>
## Day 4：LayerNorm、FFN 与 pre-LN Block

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [LayerNorm 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.LayerNorm.html) | 最后维度归一化、eps、correction=0 | 对齐均值、方差与梯度 |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 3.3 FFN 和残差/归一化图 | 组装两个子层 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读归一化与 FFN |
| 2h | 实现 LayerNorm、FFN 与 Block |
| 1h | 检查数值、梯度及残差形状 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `normalization.py`，实现沿最后 C 维的均值、方差、可学习 gamma/beta；方差用 correction=0，eps=1e-5。与相同 eps、权重的 nn.LayerNorm 比较输出、输入梯度和参数梯度。

2. 在 `blocks.py` 实现 FFN：Linear(C,4C)→GELU→Linear(4C,C)，dropout 可配置。它逐 token 处理最后的特征维。

3. 完成 pre-LN EncoderBlock，两个残差子层都保留 `[B,T,C]`，把 Mask 传递给 Attention。说明 pre-LN 与论文图中 post-LN 的区别，参考配置必须一致。

4. 创建 `tests/test_layernorm.py` 与 `tests/test_encoder.py` 中的 Block 基础检查：常量输入输出有限、batch=1 正常、归一化只沿特征维，所有参与参数可反向。完整 Encoder 检查明天继续补充。

### 分步操作与自查

#### 第一步：手算一行 LayerNorm

创建 `src/petvision/models/transformer/normalization.py`，顶层写 `import torch` 和 `from torch import nn`，定义继承 nn.Module 的 `LayerNorm(embed_dim, eps=1e-5)`，公式中的 C 即 embed_dim。再创建 `tests/test_layernorm.py`，导入 pytest、torch 和该类，将手算、常量输入及第二步的参考对照写成测试函数。

输入 `[1,3]`，均值为 2，按 C 求平均的方差为 1；gamma=1、beta=0 时输出约为 `[-1,1]`，eps=1e-5 会使数值略有偏差。

```text
mean = x.mean(dim=-1, keepdim=True)
var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
y    = (x - mean) / sqrt(var + eps) * gamma + beta
```

在自写 LayerNorm 的构造函数中，用 `self.gamma = nn.Parameter(torch.ones(C))`、`self.beta = nn.Parameter(torch.zeros(C))` 注册可学习参数；forward 使用同一组参数。普通 Tensor 属性不会自动成为模型的 parameters，不能仅写一个临时 gamma / beta。nn、torch 在 normalization.py 顶层导入。

统计范围是每个 token 的 C 个特征，不是整个 batch，也不是时间维。常量输入时归一化部分为 0，eps 防止除零。

#### 第二步：比较输出与梯度

对齐 nn.LayerNorm 的 eps、gamma、beta 与 dtype。使用相同非平凡标量目标，例如输出与固定随机 probe 点乘求和，再比较 x、gamma、beta 的梯度。

仅对归一化输出直接求和可能得到退化的零梯度例子，检查时要覆盖一般输入和一般目标。

自写 gamma/beta 分别对应参考的 weight/bias；按[Day 2 的参考检查流程](#reference-check)创建独立输入并使用同一 probe，CPU float64 的输出和三项梯度最大绝对差均以 `1e-6` 为目标。常量输入时，gamma=1、beta=0 的输出应为 0；另外改动一个 token 的特征，只应改变该 token 的归一化输出，以此发现误沿 batch 或时间维统计的问题。

#### 第三步：FFN 保留 token 数和通道出口

创建 `src/petvision/models/transformer/blocks.py`，从同包的 attention、normalization 模块导入自写 Attention 与 LayerNorm；定义 `FeedForward(embed_dim, hidden_dim, dropout=0.0)`。本轮 hidden_dim 取 4C，具体数值由创建者传入，构造函数注册两个 Linear、GELU 和可配置的 dropout，forward 返回相同 `[B,T,C]` 形状。第四步的 EncoderBlock 也写在这个文件。

FFN 的中间尺寸 `[B,T,4C]`，出口回到 `[B,T,C]`，才能残差相加。对两个不同位置的输入做扰动，FFN 本身只影响对应位置。

#### 第四步：写出 pre-LN 的实际顺序

```text
x1 = x  + attention(norm1(x), mask)
x2 = x1 + ffn(norm2(x1))
```

EncoderBlock 构造参数约定为 `embed_dim、num_heads、ffn_hidden_dim、dropout=0.0`；分别创建 norm1、Attention、norm2、FFN。forward 接收 x、attn_mask=None、key_padding_mask=None，传给 Attention 后只返回新的 x Tensor。第五周和第六周会按这个接口创建不同尺寸的新 Block。

两次归一化有各自的参数；不要误把 norm2 仍然作用在最初的 x 上。

创建 `tests/test_encoder.py`，从 `petvision.models.transformer.blocks` 导入 FeedForward 和 EncoderBlock，分别验证：FFN 输入 `[2,3,32]` 后形状不变；eval 下只修改位置 1，位置 0、2 的 FFN 输出不变；完整 Block 做一次带 Mask 的前反向，所有活跃参数梯度存在且有限。先用 dropout=0 核对数值，再用 dropout=0.1 检查 train/eval 开关。若设置 dropout，明确其放置位置并在参考对照时一起关闭：Attention 权重处、Attention 残差输出处、FFN 隐藏层及其残差输出处。

### 运行与验收

```bash
python -m pytest tests/test_layernorm.py tests/test_encoder.py -q
```

`[1,3]` 的 LayerNorm 例子应接近 `[-1,1]`；常量输入不能出现 NaN。若与 nn.LayerNorm 有系统性差异，先查是否沿最后一维求均值、方差是否按 C 求平均，以及 eps、gamma、beta 是否对齐。

- [ ] LayerNorm 输出与梯度对照通过，知道方差估计方式。
- [ ] FFN 不混合不同 token 位置，Attention 负责跨 token 信息交互。
- [ ] Block 两个残差的形状一致，Mask 传递正确。

<a id="day-5"></a>
## Day 5：位置编码与 Encoder 堆叠

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 3.5 正弦位置编码 | 理解顺序信息 |
| [TransformerEncoderLayer（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) | norm_first、batch_first、activation | 对齐参考 Block 参数 |
| [D2L 自注意力和位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html) | 正弦/余弦位置公式和形状 | 核对位置 0 与不同位置的编码 |
| [nn.Module（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) | register_buffer、state_dict、to | 让常量编码随模型迁移并被保存 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读位置编码与堆叠结构 |
| 2h | 实现位置编码和两层 Encoder |
| 1h | 组合检查与参考对照 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `positional_encoding.py`，实现正弦/余弦编码，检查偶数 C、不同 T、设备迁移；常量编码用 register_buffer 管理。

2. 组装两层 Encoder；位置编码在进入堆叠前加一次，堆叠后加一次最终 LayerNorm。EncoderBlock 保持可复用，ViT/GPT 可选择自己的可学习位置嵌入。

3. 扩展昨天的 `tests/test_encoder.py`，覆盖形状、梯度、Mask 传递、train/eval 模式和保存加载；未来隔离检查只在传入 causal mask 时要求成立。

4. 参考 nn.TransformerEncoderLayer 时对齐 norm_first=True、GELU、FFN=4C、eps、dropout=0，先对齐单 Block；完成后再检查堆叠，不用不同归一化结构作数值对照。

### 分步操作与自查

#### 第一步：按位置公式生成编码

```text
PE(pos,2i)   = sin(pos / 10000^(2i/C))
PE(pos,2i+1) = cos(pos / 10000^(2i/C))
```

本周只支持偶数 C。位置 0 的偶数通道为 0、奇数通道为 1，可以作最小数值检查；不同 token 位置应获得不同编码。

创建 `src/petvision/models/transformer/positional_encoding.py`，顶层写 `import torch` 和 `from torch import nn`，实现 `SinusoidalPositionalEncoding(embed_dim, max_length=128)`。先用公式构造 `[1,max_length,C]` 的 table，再在构造函数调用 `self.register_buffer("pe", table)`；forward(x) 检查尺寸后返回 `x + self.pe[:, :x.shape[1]]`。它本身是 nn.Module，但 pe 作为常量 buffer 不参与梯度更新。

Encoder 构造时创建 `self.position = SinusoidalPositionalEncoding(...)`，输入转到 GPU 后，buffer 随整个模型迁移。

构造时约定最大位置长度，例如 max_length=128，保存形状 `[1,max_length,C]` 的 buffer；forward 检查 C 为偶数、T 不超过上限，再取 `[:, :T]`。用 T=1、T=8 验证广播，用奇数 C 和越界 T 验证明确报错；将整个模型 `.to(device=..., dtype=...)` 后，核对 buffer 与输入的 dtype/device 相同。位置表属于 buffer，Block 的可学习参数仍通过 parameters 管理。

#### 第二步：堆叠时正确传递同一 Mask

在已有 `src/petvision/models/transformer/blocks.py` 中导入位置编码类并增加 `Encoder(embed_dim, num_heads, ffn_hidden_dim, depth=2, dropout=0.0, max_length=128)`，由它创建位置编码、独立 Block 实例和最终 LayerNorm。下面片段放在 Encoder.forward 内，最后还要执行最终 LayerNorm 并返回结果。

```python
x = self.position(x)
for block in self.blocks:
    x = block(x, attn_mask=attn_mask,
              key_padding_mask=key_padding_mask)
```

Encoder 入口加一次位置编码，不能每过一层再重复加同一份。可学习位置嵌入留给 ViT/GPT 按各自结构选择。

保持 EncoderBlock 本身不添加位置编码、不执行堆叠后的最终 LayerNorm；它只负责自己的 pre-LN Attention/FFN 和残差。教学 Encoder 负责正弦位置编码、Block 堆叠与最终归一化；第五周 ViT 和第六周 GPT 可复用 Block 类，分别创建新实例并管理各自的位置参数与最终归一化。

#### 第三步：先对齐一层，再对齐多层

扩展 `tests/test_encoder.py`，导入 Encoder 和 SinusoidalPositionalEncoding；分别添加位置编码、单 Block 参考对照、两层堆叠、Mask 传递和保存加载测试，不覆盖昨天的 FFN / Block 检查。

参考 EncoderLayer 对齐 batch_first、norm_first、GELU、FFN 宽度、eps 和 dropout。先完成单 Block 权重映射与数值比较，再检查两层组合。

单 Block 的映射按模块列清楚：Attention 的 QKV/out_proj 对应 `reference.self_attn`，两个归一化对应 norm1/norm2，FFN 两个 Linear 对应 linear1/linear2；gamma/beta 对应 weight/bias。先确认所有参数均有对应项，再沿用 Day 2 的相同 probe 检查输出、输入梯度与映射后的参数梯度，CPU float64 容差 `1e-6`。参考 Block 不包含入口位置编码或堆叠后的最终 LayerNorm，这两项由外层 Encoder 单独检查。

检查不同 T 是否正常、输入输出是否保持 `[B,T,C]`，以及 Mask 是否真的传到每一层。默认双向 Encoder 不要求未来隔离；传 causal 时才检查。

#### 第四步：保存加载仍保留位置参数或 buffer

重建模型加载后比较固定输入；确认 state_dict 包含设计中需要保存的 buffer，位置长度和模型配置可以恢复。

两层用 `ModuleList([EncoderBlock(...), EncoderBlock(...)])` 分别创建，检查对应参数不是同一个对象，避免 `[block] * 2` 意外共享权重。把 Day 3 的前缀扰动检查用于完整 Encoder，确认 causal Mask 经过每一层；再以 dropout=0.1 切 train/eval，检查子层 training 标记随之改变，eval 下固定输入重复输出一致。保存加载使用第二周的 tmp_path 方法，加载后核对位置 buffer 和固定输入输出，最大差小于 `1e-6`。

### 运行与验收

```bash
python -m pytest tests/test_encoder.py -q
```

检查位置 0 的正弦/余弦值、不同 T 的输出、两层前反向与保存加载；输入 `[2,8,32]` 时输出仍为 `[2,8,32]`。若 GPU 上出现设备不一致，检查位置编码是否注册为 buffer；若只有多层的因果检查失败，检查每一层是否都收到了同一 Mask。

- [ ] 位置编码形状、dtype 与 device 匹配输入。
- [ ] 两层参数独立、前反向正常，Mask 不会在堆叠过程中丢失；模式切换和包含位置 buffer 的保存加载检查通过。
- [ ] 知道位置嵌入的加入位置及各归一化位置。

<a id="day-6"></a>
## Day 6：统一检查、张量流与有限参考重写

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 按自己的模块回看 Encoder 部分 | 把图和代码一一对应 |
| [Attention 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.MultiheadAttention.html) | 复核权重与 Mask 接口 | 完成最终参考检查 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 运行所有本周关键检查 |
| 1.5h | 画张量流与解释问题 |
| 1h | 有限参考重写多头核心 |
| 0.5h | 整理本周 README |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 运行 Attention、LayerNorm、Encoder 的全部检查；保留真实最大误差和检查命令在周 README，不额外为每个检查生成 CSV。

2. 创建 `experiments/transformer_encoder/week4/tensor_flow.md`，画 `[B,T,C]→QKV→[B,h,T,d]→[B,h,T,T]→[B,T,C]`，标注 Mask、softmax、残差和 FFN；这是一份模块说明，不是每日记录。

3. 只看公式重写多头拆分、缩放、Mask 与合并的核心片段；口头解释头数变化与 head_dim 的关系。

4. 确认第五周能导入 EncoderBlock 来组装 ViT，第六周能向 Block 传入 causal mask；周总结记录检查、形状图和仍需理解的点。完整 Encoder–Decoder 与复制/反转训练保留为拓展。

### 分步操作与自查

#### 第一步：把每项检查对应到一个错误

形状检查发现轴和拼接错误；归一化检查发现 softmax 维度错误；未来扰动检查发现 causal 泄漏；参考梯度检查发现计算路径错误。

在 README 中记录各检查的命令、输入配置、最大误差和是否通过，不额外为这些小检查创建指标 CSV。

#### 第二步：画一个具体尺寸的完整张量流

创建 `experiments/transformer_encoder/week4/tensor_flow.md`，将下面的流程图与形状说明写进该文件，并在本周 README 添加相对链接 `tensor_flow.md`。

使用 B=2、T=8、C=32、heads=4，把 QKV、scores、weights、合并输出和两个残差标到 tensor_flow.md。旁边标注 mask 在 softmax 前应用、LayerNorm 沿 C。

#### 第三步：有限参考重写核心片段

只保留公式，重写投影→拆头→缩放点积→屏蔽→softmax→加权 V→合并。用已知小例子检查，不要求再抄一整套工程文件。

实际调用草稿中的重写片段：先复现零 query 得到 `[3,2]`，再用 B=2、T=3、C=8、heads=2 核对拆头/合并形状与行和；最后加 T=3 的 causal Mask，检查未来位置权重为 0。原模块检查通过之外，保留这些直接运行重写片段的输出或断言。

#### 第四步：确认下周接口可直接使用

ViT 需要双向 Encoder、变长 token 数；GPT 需要同样的多头核心和 causal mask。确认代码没有把 T 写死，也没有在导入时训练或打印演示。

本周完整 Encoder–Decoder、Cross-Attention 训练与序列复制任务仍在拓展范围；当前验收聚焦已实现模块与解释。

### 运行与验收

```bash
python -m pytest tests/test_attention.py tests/test_layernorm.py tests/test_encoder.py -q
```

全部检查通过后，核对 `tensor_flow.md` 能解释实际代码中的每个主要形状，并在本周 README 用几行汇总配置、实际误差、运行命令和遗留问题。今天不需要额外训练分类器或语言模型。

- [ ] 形状、有限梯度、归一化、Mask 和参考输出/梯度检查齐全。
- [ ] 核心模块可独立导入，导入时不运行演示。
- [ ] 能解释 Q/K/V、缩放、位置编码、LayerNorm、FFN 与残差。

## 本周结束检查

- [ ] 多头 Attention 由基础算子手写，官方层仅用作对照。
- [ ] 有效 query 权重归一化、padding 屏蔽和未来隔离检查通过。
- [ ] LayerNorm 与 Attention 的输出及梯度与参考一致。
- [ ] Encoder、张量流说明与每周 README 可供 ViT 和 GPT 复用。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第五周](week5.md)。

