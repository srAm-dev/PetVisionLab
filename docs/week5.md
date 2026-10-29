# 第五周逐日学习指南：小型 Vision Transformer 与视觉模型比较

> 更新日期：2026-10-27<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-2)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：复用 Encoder 实现 ViT，完成 Patch=4/8 对照，并在配置冻结后统一测试 CNN、ResNet 和 ViT，形成带局限说明的视觉报告。  
> 环境：本地 Conda `petvision`、RTX 4060；沿用 PyTorch、Matplotlib、PyYAML、pytest。首轮 FP32，小模型和 batch=64 起步。

## 怎么使用这份指南

先确认[上一周指南](week4.md)的基础关卡已通过。第四周 Attention / Mask / LayerNorm 检查需通过；第三周的 CIFAR 固定划分、四组配置和 best checkpoint 应能复用。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/vit_patch_size/week5/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── src/petvision/models/cv/vit.py           # Day 1–2：Patch、CLS、ViT
├── src/petvision/evaluation/classification.py # Day 4：F1、混淆矩阵、计时
├── configs/cv/
│   ├── vit_cifar10.yaml                   # Day 2：Patch=4
│   └── vit_patch8_cifar10.yaml             # Day 3：Patch=8
├── scripts/train_cv.py                     # Day 2–3：新增 ViT 选择，运行两组实验
├── scripts/evaluate.py                     # Day 4–5：统一指标
├── tests/test_vit.py                       # Day 1–2：Patch、位置与反向
├── experiments/vit_patch_size/week5/
│   ├── README.md                         # 已提供：运行入口与最终报告
│   ├── patch_results.csv                 # Day 4：两组验证对照
│   ├── visual_test_results.csv           # Day 5：六个冻结变体测试
│   ├── runs/{patch4,patch8}/              # 配置、history、run_summary 与各组评估结果
│   └── figures/                          # 跨模型对照曲线；单模型图在各自 experiment_dir/figures
└── outputs/{vit_patch4,vit_patch8}/
    └── {best,last}.pt                     # 实际训练生成
```

## 开始前：激活环境并检查已有模块

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

解释器应位于 Conda 的 `petvision` 环境，`petvision.__file__` 应指向当前仓库的 `src/petvision/`。若出现 `No module named petvision`，先确认第二周已建立 `pyproject.toml` 和包目录，再在根目录执行 `python -m pip install -e . --no-deps`；若缺少 torch 或 CUDA 不可用，按[第二周 Day 1 的安装与排错步骤](week2.md#environment)逐项核对。

复用第三周 CIFAR-10 固定划分与第四周的手写 EncoderBlock。 本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

<a id="vit-shapes"></a>
ViT（Vision Transformer）先把图片切成小块，再把每个小块转换为一个特征向量，作为 Transformer 的输入。下面的符号与第三周卷积公式有区别：

| 符号 / 参数 | 含义 | 本周例子 |
| --- | --- | --- |
| B、H、W | 批次图片数、输入高度、输入宽度 | `[B,3,32,32]` 中 H=W=32 |
| P / patch_size | 每块 Patch 的边长，不是 padding | P=4 表示每块 4×4 像素 |
| N | 一张图片切出的 Patch 数，不是数据集样本数 | `N=(H/P)*(W/P)`，P=4 时为 64 |
| C / embed_dim | 一个 Patch 投影后的特征数 | 128；输入 RGB 通道数仍为 3 |
| CLS | 可学习的分类汇总 token | 加在 Patch 序列最前面，占一个位置 |
| T | 进入 Attention 的序列长度 | `T=N+1`，包含 CLS |
| depth、heads、ffn_hidden_dim | Block 层数、注意力头数、FFN 中间宽度 | 2、4、512；每头维度为 128/4=32 |

沿用第三周的 `cifar10_split.json`、预处理与训练/验证边界。小 ViT 起步 C=128、depth=2、heads=4、FFN=512、dropout=0.1、CLS 分类、可学习位置嵌入；两个 Patch 版本都从零初始化。AdamW lr=3e-4、weight_decay=0.01、batch=64、20 epochs、seed=42 为首轮预算，先实测短运行再固定。

32×32 图片：Patch=4 得到 64 个图像 token、含 CLS 长度 65；Patch=8 得到 16 个图像 token、含 CLS 长度 17。两组仅改变 Patch 相关结构，C、层数、划分、增强与训练预算一致；Patch 投影与位置参数量会改变，需记录。ViT 使用双向 Attention，不启用 causal mask。跨 CNN/ResNet/ViT 比较允许优化配置不同，但必须列出差异，不把准确率差异全归因于架构。

这里“相同训练预算”具体指相同 epoch、batch 与样本数，并不表示相同计算量或耗时。每头 Attention 分数矩阵从 65×65 变为 17×17，元素数约相差 14.6 倍；整个模型还包含投影、FFN 等计算，不能据此断言训练就快 14.6 倍。报告应同时看验证指标、参数量和实测时间；某个 Patch 更优是本次设置下的观察，不是验收必须得到的结论。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | Patch Embedding 与位置形状 | 4h | 1h | 两种 Patch 的可检查实现 |
| Day 2 | 复用 Encoder 组装 ViT | 4h | 1h | 可训练 ViT 与小批过拟合 |
| Day 3 | Patch=4/8 正式对照 | 4h | 1h | 两个训练结果与曲线 |
| Day 4 | 统一指标与冻结视觉配置 | 4h | 1h | 验证对照表和统一评估函数 |
| Day 5 | 统一 CIFAR-10 测试与错误分析 | 4h | 1h | 六个变体的最终视觉结果表 |
| Day 6 | 视觉报告与核心流程重写 | 4h | 1h | 分类比较报告及可复现材料 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：Patch Embedding 与位置形状

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [ViT 原论文](https://arxiv.org/abs/2010.11929) | 第 3 节、Figure 1 与图像 Patch 表达 | 手算 token 数和展开顺序 |
| [torchvision ViT 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html) | patch projection、CLS 与位置嵌入 | 作为结构参考 |
| [F.unfold（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.functional.unfold.html) | kernel_size、stride、输出的 Patch 维度 | 与 Conv 投影做独立数值对照 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读 ViT 结构并手算尺寸 |
| 2h | 实现 Patch 投影与顺序检查 |
| 1h | 检查 Patch=4/8 和非法尺寸 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `src/petvision/models/cv/vit.py` 的 PatchEmbedding，使用 Conv2d(kernel_size=P,stride=P) 后 flatten 空间维并转置到 `[B,N,C]`，或显式展开后 Linear，明确选用哪一种。

2. 用含位置编号的 4×4 小图手算 P=2 的四个 Patch 顺序；若使用 Conv 形式，对齐权重到显式 unfold 版本验证。

3. 创建 `tests/test_vit.py`，检查 32×32 的 P=4/8 分别得到 N=64/16；H/W 不能被 P 整除时明确报错。

4. 添加可学习 CLS `[1,1,C]` 与位置嵌入 `[1,N+1,C]`，CLS 按 batch 扩展；位置长度不匹配时不要静默截断。

### 分步操作与自查

#### 第一步：把 Patch 数与每个 Patch 的维度分开

创建 `src/petvision/models/cv/vit.py`，顶层写 `import torch` 和 `from torch import nn`，定义继承 nn.Module 的 `PatchEmbedding(in_channels, patch_size, embed_dim)`。选择 Conv2d 或显式展开后 Linear 的一种正式路径，forward 返回 `[B,N,C]`；下方两条形状用于理解同一投影过程。今天还在同一文件建立 ViT 的 CLS / 位置参数，完整 ViT.forward 明天接通。

对输入 `[B,3,32,32]`，P=4 时每个 Patch 含 `3*4*4=48` 个数，共 `8*8=64` 个 Patch；P=8 时维度为 192，共 16 个 Patch。

以 P=4 为例，高度方向能放 `32/4=8` 块，宽度也能放 8 块，所以共 64 块；每块包含 3 个通道各 4×4 个像素，因此展开长度为 48。Linear(48,128) 把每块的 48 个数投影为 128 个特征，不改变 64 这个块数。若用 Conv2d，设置 in_channels=3、out_channels=128、kernel_size=4、stride=4、padding=0，也得到同样的块数与特征数。

```text
P=4：图像 → [B,64,48]  → Linear → [B,64,128]
P=8：图像 → [B,16,192] → Linear → [B,16,128]
```

投影后的 C 都为 128；不要把 Patch 像素维度直接当作 Encoder hidden size。

#### 第二步：用小图确认展开顺序

创建 `tests/test_vit.py`，顶层导入 pytest、torch 和 `from petvision.models.cv.vit import PatchEmbedding`。将编号图、P=4/8 的形状、投影值对照和非法尺寸检查分别写成测试函数；第二天继续向同一文件添加完整 ViT 检查。

对 4×4 单通道编号图，P=2 的第一个 Patch 应为左上角 2×2，第二个为右上角，之后按行进入下一排。

```text
原图：                 四个 Patch 按顺序展平：
 0  1  2  3            [0,1,4,5]
 4  5  6  7            [2,3,6,7]
 8  9 10 11            [8,9,12,13]
12 13 14 15            [10,11,14,15]
```

这张小图只检查单通道 Patch 顺序；真正的 CIFAR 输入有三个通道。对照 Conv 和显式展开版本时，通道与 Patch 内像素的排列也需要一致。

用 Conv2d 投影时先得到 `[B,C,H/P,W/P]`，再 flatten(2).transpose(1,2)。与显式 unfold 版本对照时，权重和 Patch 顺序也要对齐。

这里 `flatten(2)` 从第 2 轴开始合并高宽两轴，例如 `[B,128,8,8]→[B,128,64]`；`transpose(1,2)` 再交换特征轴和 Patch 轴，得到 `[B,64,128]`。位置编号顺序沿每一行从左到右，再进入下一行。

把「对齐」落实为一个小张量检查：先在 CPU float64 上建立相同输出通道 C、bias=True 的 Conv2d 与 Linear；Linear 输入宽度为 `输入通道数*P*P`。给定同一图片后：

```python
patches = torch.nn.functional.unfold(images, kernel_size=P, stride=P).transpose(1, 2)
with torch.no_grad():
    linear.weight.copy_(conv.weight.flatten(1))
    linear.bias.copy_(conv.bias)
conv_tokens = conv(images).flatten(2).transpose(1, 2)
linear_tokens = linear(patches)
assert (conv_tokens - linear_tokens).abs().max().item() < 1e-6
```

先检查单通道编号图的四个原始 Patch，再用三通道随机图片核对投影值，避免只看形状就漏掉通道/像素顺序错误。非法尺寸在 forward 开始时检查 `H % P`、`W % P`，用 `pytest.raises(ValueError)` 覆盖。

#### 第三步：加入 CLS 和位置参数

在 `src/petvision/models/cv/vit.py` 定义 `ViT` 类，构造参数约定为 num_classes、image_size、patch_size、embed_dim、depth、heads、ffn_hidden_dim、dropout；先创建 PatchEmbedding、CLS 和位置参数，Day 2 再补 Block 与分类头。PatchEmbedding 本身只负责图片投影，不把 CLS / 位置嵌入也藏进它的输出。

```text
Patch tokens       [B,N,C]
CLS expand         [B,1,C]
concat             [B,N+1,C]
position embedding [1,N+1,C]
```

CLS 是所有样本共享的一组可学习参数，扩展到 batch 后参与训练；它不是预先算出的类别标签。位置 0 对应 CLS，后面才是图像 Patch。

构造函数中用 `nn.Parameter` 注册 CLS 和位置嵌入，例如先以 `torch.zeros(1,1,C)` 和 `torch.zeros(1,N+1,C)` 建立 Parameter，再各用 `nn.init.normal_(parameter, mean=0.0, std=0.02)` 初始化；记录这一初始化方式。forward 只 expand/concat/相加，不临时创建新 Parameter。检查两者出现在 named_parameters 和 state_dict 中，位置长度与实际 N+1 不符时明确报错，B=1 与 B=4 都保持 batch 维。

Day 1 尚未接好完整 ViT.forward 时，在测试里明确调用 patch_embed，手动 expand / concat / 加位置参数，核对这些中间结果即可；不要今天就调用尚未实现的完整分类前向。构造 ViT 时先检查 image_size 能被正整数 patch_size 整除；完整 forward 还要检查输入 H、W 均等于配置 image_size。仅比较 token 数不能识别例如 16×64 与 32×32 这类同数量、不同位置网格的输入。

### 运行与验收

```bash
python -m pytest tests/test_vit.py -q
```

今天先检查 Patch、CLS 和位置尺寸，完整模型明天补充。P4/P8 的 Patch 输出应分别为 `[B,64,128]`、`[B,16,128]`，加 CLS 后为 `[B,65,128]`、`[B,17,128]`。若顺序检查失败，先打印编号小图的四个 Patch。

- [ ] Patch 数、顺序与投影输入维度清楚。
- [ ] 含 CLS 的序列长度为 65/17，位置嵌入匹配。
- [ ] 形状检查支持 batch=1 和 batch>1。

<a id="day-2"></a>
## Day 2：复用 Encoder 组装 ViT

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [ViT 原论文](https://arxiv.org/abs/2010.11929) | Encoder、CLS 分类和训练设置说明 | 组装最小分类模型 |
| [第四周指南](week4.md) | pre-LN Encoder 接口与检查 | 复用已验证模块 |
| [Python 计时（官方中文）](https://docs.python.org/zh-cn/3.12/library/time.html#time.perf_counter)、[CUDA 同步（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.cuda.synchronize.html) | perf_counter、GPU 异步与同步；显存接口按下方说明使用 | 记录小模型真实耗时与峰值显存 |
| [重置显存峰值](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html)、[读取分配峰值](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)（官方英文） | reset_peak_memory_stats、max_memory_allocated 与字节单位 | 明确峰值的统计起点和范围 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 确认可复用 Encoder 的接口 |
| 2h | 组装 ViT 与训练配置 |
| 1h | 形状、反向、smoke 与过拟合 |
| 0.5h | 测时并固定首轮预算 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 完成 ViT：Patch→CLS/可学习位置→两层 pre-LN EncoderBlock→最终 LayerNorm→CLS 分类头，输出 `[B,10]`。复用第四周的 Block 类并创建本模型的新实例，保留手写 Attention 和 FFN。

2. 创建 `configs/cv/vit_cifar10.yaml`，在训练和独立评估入口的模型选择处添加 ViT。记录第三周划分文件路径，训练和验证实例分别加载正确预处理。

3. 补充 `tests/test_vit.py` 的前反向、CLS、位置参数梯度与保存加载检查；推理用 model.eval()。

4. 先 smoke，再固定 64 样本过拟合，记录一轮耗时与峰值显存。预算不足时在正式训练前统一调整两种 Patch 的 epoch/batch，并写明原因。

### 分步操作与自查

<a id="encoder-reuse"></a>
#### 第一步：明确 ViT.forward 与复用模块的边界

继续修改 `src/petvision/models/cv/vit.py`，从 `petvision.models.transformer.blocks` 导入 EncoderBlock，从 `petvision.models.transformer.normalization` 导入 LayerNorm，在昨天的 ViT 类中补齐构造函数和 forward；模型定义保留在该文件，训练入口只负责创建和调用它。

```python
def forward(self, images):
    # Patch → CLS/可学习位置 → 两个 EncoderBlock → norm → 分类 logits
    ...
```

网络只返回 `[B,10]`，训练损失、优化器和数据 loader 继续留在训练器。不要在 forward 里额外对类别做 softmax。

推荐直接复用第四周 `EncoderBlock` 的实现，用 ModuleList 创建两个新的 Block；ViT 自己负责 CLS、可学习位置嵌入与堆叠后的最终 LayerNorm。第四周的教学 Encoder 还包含正弦位置编码和最终归一化，不能直接嵌入后再把同样的职责做一遍。

| 计算位置 | ViT 中由谁负责 | 次数 |
| --- | --- | --- |
| CLS 拼接、可学习位置相加 | ViT | 进入 Block 堆叠前各一次 |
| norm1、Attention、norm2、FFN 与两个残差 | 各个 EncoderBlock | 每层各自执行 |
| 堆叠后的最终 LayerNorm、取 CLS、分类头 | ViT | 各一次 |

整体顺序可先写为以下核心片段；构造函数中的 `self.blocks` 使用两个新建的 EncoderBlock，`self.final_norm` 是最终 LayerNorm：

构造函数实际按 depth 创建 `nn.ModuleList`，每次循环新建 `EncoderBlock(embed_dim, heads, ffn_hidden_dim, dropout)`，不写死为两个共享对象。final_norm 使用 LayerNorm(embed_dim)，head 使用 nn.Linear(embed_dim, num_classes)；下方代码位于 ViT.forward 的输入尺寸检查之后，结尾还需 `return logits`。

```python
tokens = self.patch_embed(images)
cls = self.cls_token.expand(tokens.shape[0], -1, -1)
tokens = torch.cat((cls, tokens), dim=1)
tokens = tokens + self.position_embedding
for block in self.blocks:
    tokens = block(tokens, attn_mask=None, key_padding_mask=None)
tokens = self.final_norm(tokens)
logits = self.head(tokens[:, 0])
```

若选择封装后的完整 Encoder，需要显式关闭其内部正弦位置编码，并约定最终 LayerNorm 只由一处执行；本周按上面的 Block 组合方式即可。自查时看一次 forward 的调用顺序，确认位置只加一次、最终 LayerNorm 只有一次；Block 内部的 norm1/norm2 仍照常保留。

#### 第二步：逐层检查尺寸

P=4 时序列长 65，P=8 时长 17，Encoder 入口和出口均保持 `[B,T,128]`。最终 LayerNorm 后取 `x[:,0]` 得到 `[B,128]`，分类头输出 `[B,10]`。

在 `tests/test_vit.py` 中按这个顺序完成检查，调试时可临时分段调用子模块：

1. 创建 P=4、B=2 的随机输入，先只调用 patch_embed，确认得到 `[2,64,128]`；若这里不对，先检查卷积输出和 flatten / transpose。
2. 将 CLS 扩展到 `[2,1,128]`，拼接在第 0 个位置，确认长度从 64 变为 65；再核对位置参数为 `[1,65,128]`，通过 batch 广播相加。
3. 依次调用两个 Block，分别确认入口与出口的 T 都是 65，并确认没有传 causal mask；不要只检查最后一个 Block。
4. 执行最终 LayerNorm，再取第 0 个位置送入分类头；确认取出的是 `[2,128]`，而不是把全部 65 个位置都送去作为独立分类结果。
5. 再新建 P=8 的模型，从头重复检查，此时 Patch 数为 16、含 CLS 长度为 17。不能只替换输入而沿用 P4 的位置参数。
6. 将两种模型都改用 B=1 输入再运行，确认 logits 仍为 `[1,10]`。通过后移除临时打印，保留形状断言，进入反向检查。

ViT 使用双向 Attention；若沿用 GPT 的 causal 开关，CLS 位于序列最前时将看不到后面的图像 Patch，这是需要检查的结构错误。

#### 第三步：验证参数确实参与学习

扩展已有 `tests/test_vit.py`，再导入 ViT；本步的 B=1 前反向、参数梯度与保存加载检查写入这个文件，保留昨天的 Patch 检查。64 样本拟合通过第四步接好后的训练入口运行。

对随机图像和标签做一次 backward，查看 Patch 投影、CLS、位置嵌入、Encoder 和 head 的梯度是否存在且有限。B=1 也要通过。

再对固定 64 条训练图做过拟合，训练时关闭随机增强以减少干扰。loss 不降时先检查取出的是否为 CLS、模型是否包含参数、优化器是否覆盖参数。

保存加载检查使用一个固定随机图片批次、eval 和第二周的 tmp_path 流程，新模型加载后 logits 最大差小于 `1e-6`，同时核对 CLS/位置参数。测一轮训练前在 CUDA 分支调用 `torch.cuda.synchronize()`、`torch.cuda.reset_peak_memory_stats()`，用 `time.perf_counter()` 记开始/结束，结束前再次同步；峰值取 `torch.cuda.max_memory_allocated()/1024**2`，注明 MiB 和测量范围。CPU 分支跳过 CUDA 接口，记录设备即可。

正式运行沿用[第三周逐轮追加与资源保存](week3.md#training-resources)：每组的 experiment_dir 同时保存 history.csv、run_summary.json 和配置快照。测时前重置峰值，计时范围覆盖训练 / 验证 / 逐轮保存；短检查的数字只用于确定预算，不写成正式两组训练时间。

#### 第四步：接入已有训练入口

修改已有 `scripts/train_cv.py` 和 `scripts/evaluate.py`，顶层从 `petvision.models.cv.vit` 导入 ViT，在已有模型选择处添加 vit 分支，按下方 model 字段创建该类。两处构造参数保持一致，独立评估继续按 checkpoint 保存的 model_config 重建，具体职责见[第三周入口整理步骤](week3.md#cv-entry-refactor)。

模型选择添加 vit，配置明确 image_size、patch_size、embed_dim、depth、heads、dropout；split_path 原样复用第三周文件。通过短运行后再冻结预算。

创建 `configs/cv/vit_cifar10.yaml`。数据、增强与归一化延续第三周的 [v2 预处理](week3.md#cifar-v2-transforms)：训练入口和评估入口复用已有 CIFAR 分支及 `build_cifar10_loaders`，无需为 ViT 另建一套 transform。训练顺序为 v2.ToImage → v2.RandomCrop → v2.RandomHorizontalFlip → v2.ToDtype(torch.float32,scale=True) → v2.Normalize；验证和测试去掉两种随机增强。mean/std、裁剪 padding 和翻转开关继续读取下方 data 配置，输出为 float32 的 `[B,3,32,32]`，再由 ViT 切分 Patch。以下为完整起始配置：

```yaml
model:
  name: vit
  num_classes: 10
  image_size: 32
  patch_size: 4
  embed_dim: 128
  depth: 2
  heads: 4
  ffn_hidden_dim: 512
  dropout: 0.1
data:
  name: cifar10
  root: datasets/raw
  split_path: experiments/cnn_vs_resnet/week3/cifar10_split.json
  validation_size: 5000
  image_size: 32
  batch_size: 64
  num_workers: 0
  augmentation:
    random_crop_padding: 4
    random_horizontal_flip: true
  normalize:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
training:
  seed: 42
  device: auto
  dtype: float32
  epochs: 20
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
experiment_dir: experiments/vit_patch_size/week5/runs/patch4
output_dir: outputs/vit_patch4
```

`model.image_size` 用于建立 Patch 和位置参数，`data.image_size` 对应实际输入，构建时核对两者都为 32。heads、FFN 宽度与 dropout 都要传入新建的 Block，不能继续使用第四周演示时写死的 C=32。

### 运行与验收

```bash
python -m pytest tests/test_vit.py -q
python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --smoke
python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --overfit-samples 64
```

确认输出为 `[B,10]`，再检查固定 64 条训练图的 loss 能否下降。若 CLS 没有梯度，检查它是否注册为 Parameter、是否参与 concat、分类头是否取 `tokens[:,0]`；无法学习时也要检查是否误用了 causal mask。今天完成短检查与测时，正式训练在 Day 3 执行。

- [ ] 前向与反向正常，CLS 和位置参数参与学习。
- [ ] 复用 Block 类创建新模型，未重复加入正弦位置编码或堆叠后的最终 LayerNorm。
- [ ] 小批数据能够拟合，训练器接口与 CNN/ResNet 兼容。
- [ ] 两组训练预算在比较前固定。

<a id="day-3"></a>
## Day 3：Patch=4/8 正式对照

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [ViT 原论文](https://arxiv.org/abs/2010.11929) | Patch 表示与计算代价相关部分 | 解释序列长度变化 |
| [Attention 论文](https://arxiv.org/abs/1706.03762) | 注意力矩阵与序列长度 | 比较 T×T 的开销 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 列出 Patch 对照控制条件 |
| 2h | 完成配置并运行两组正式训练 |
| 1h | 核对指标、参数量和耗时 |
| 0.5h | 整理验证曲线与配置快照 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `vit_patch8_cifar10.yaml`，与 Patch=4 使用独立 experiment_dir 和 output_dir；两组均从新模型训练，不能在另一 Patch checkpoint 上直接继续。

2. 固定 C、depth、heads、FFN、dropout、优化器、epoch、batch、增强、seed 和划分，只改变 Patch 及随之改变的投影/位置形状。

3. 每组记录 train/val loss、accuracy、best epoch、参数量、实际训练时间与峰值显存，保存 best/last。

4. 画两组验证曲线，解释 attention 矩阵元素量从 65² 到 17² 的变化；实际总耗时还受投影、FFN 与硬件利用率影响，直接记录实测值。

### 分步操作与自查

#### 第一步：把不可避免的结构变化写清楚

Patch 大小变化会同时改变投影参数量和位置长度。以 Conv 投影 bias=True、C=128 为例，P4 投影参数为 6272，P8 为 24704；位置参数分别为 8320 和 2176。

这些数按“张量中的标量元素数”计算：投影 weight 为 `[C,3,P,P]`，bias 为 `[C]`，因此共有 `C*3*P*P+C` 个参数；位置嵌入为 `[1,N+1,C]`，共有 `(N+1)*C` 个参数。代入 P=4：`128*3*4*4+128=6272`、`65*128=8320`；P=8：`128*3*8*8+128=24704`、`17*128=2176`。CLS 自身两组都为 128 个参数，不造成差值。

因此 token 更少不等于总参数一定更少；分类头与 Encoder 不变时，本例 P8 总参数比 P4 多 12288。以自己的实际实现统计为准。

#### 第二步：比较计算规模与实测耗时

```text
P4 attention 每头矩阵：65 × 65
P8 attention 每头矩阵：17 × 17
```

矩阵元素数比约 14.6，但整网还有投影、FFN、数据加载和硬件开销，不能直接把这个比值当作实际训练加速比。

#### 第三步：固定可比条件后分别训练

首轮可固定各自 seed 独立初始化，并记录这一方式；相同 seed 不保证公共层的初值相同，因为 Patch 层消耗的随机数数量不同。若选择对齐公共层，只从 step=0 状态复制 Encoder、最终 LayerNorm、分类头及同形状的 CLS，Patch 投影和位置嵌入独立初始化。数据划分、增强策略、采样种子和训练预算保持一致。

复制完整 P4 配置到 `configs/cv/vit_patch8_cifar10.yaml`，将 `model.patch_size` 改为 8，experiment_dir 改为 `experiments/vit_patch_size/week5/runs/patch8`，output_dir 改为 `outputs/vit_patch8`。若对齐公共层，在正式第一次前向之前保存 P4 的 initial.pt，并为 P8 明确实现允许复制的键名范围；逐项检查这些键的 shape 和复制后数值。第三周加载规则要求所有共有键形状一致，而本周 Patch 与位置参数同名但形状不同，不能直接套用。显式排除这两类参数并记录差异，其他意外不匹配仍报错；不能使用训练后的 best。

每组从新优化器开始，在独立 experiment_dir 保存 history 和配置，在独立 output_dir 保存 checkpoint。沿用第二周规则，best 按验证 accuracy 选择，持平保留较早一次，last 保存最后一轮。

数据顺序使用独立 DataLoader Generator；两组 token 长度不同，dropout 会消耗不同数量的随机数，相同全局 seed 不保证逐步 dropout 或增强结果相同。记录相同策略与种子即可；只有独立控制增强随机流或保存增强决策后，才能声称每张图的随机增强完全一致。单种子对照的结论限定在本次设置。

#### 第四步：读曲线而非只看最后一格

1. 读回两个 history.csv，检查 epoch 连续、没有重复，两组都达到冻结的预算。先转换数值列，再绘图，避免 CSV 字符串被当作类别坐标。
2. 分别找到两组最佳验证 accuracy 所在行，记录该行 epoch、val_loss 与 val_accuracy；同组的两个指标必须取自同一行。
3. 用相同横轴画两组 train/val 曲线。先看训练是否都在下降，再看验证是否同步改善；若验证后期转差，回看该组 best 是否早于 last。
4. 打开各组 run_summary.json，将实际训练秒数和参数量与曲线放在一起解释。相同 epoch 下较快不等于相同时间预算下准确率更高。
5. 在 README 写出一个观察和一个限制，例如哪一组在本次预算下验证更好，以及单种子、参数量变化是否限制归因。没有完成两组时先标记待比较，不填预期结论。

读取 patch4、patch8 各自的 history.csv，在同一张图上画两组 train/val 曲线，使用第一周的绘图与保存方法写入 `experiments/vit_patch_size/week5/figures/patch_comparison.png`，父目录先创建；在本周 README 链接这张图和两份配置。

### 运行与验收

```bash
python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml
python scripts/train_cv.py --config configs/cv/vit_patch8_cifar10.yaml
```

依次运行两组，从各自新模型和新优化器开始。分别检查 `runs/patch4`、`runs/patch8` 的配置和 history，以及 `outputs/vit_patch4`、`outputs/vit_patch8` 的 best/last。显存不足时，在正式对照前统一调整两组预算，并记录实际值。

- [ ] 两组完成相同预算，结果与配置一一对应。
- [ ] token 数、参数量和耗时差异有真实记录。
- [ ] 当前仍按验证集分析，测试集尚未用于挑选配置。

<a id="day-4"></a>
## Day 4：统一指标与冻结视觉配置

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 分类教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/cifar10_tutorial.html) | 逐类统计与预测结果 | 扩展为混淆矩阵与 Macro-F1 |
| [总计划结果模板](StudyPlan.md#acceptance) | 视觉结果列与比较边界 | 冻结六个变体配置 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 复核六个变体的配置与 checkpoint |
| 2h | 实现混淆矩阵、Macro-F1 和计时 |
| 1h | 整理验证表并冻结最终评估设置 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `src/petvision/evaluation/classification.py`，从真实类别和预测统计 10×10 混淆矩阵，计算 accuracy 和逐类 F1 后取 10 类平均；分母为零时按约定记 0。

2. 扩展 `evaluate.py`，输出 split、样本数、loss、accuracy、macro_f1、参数量和测量配置；用手写小预测例子核对指标实现。

3. 计时固定相同 device、dtype、batch、输入尺寸，先预热；GPU 计时前后同步。把加载时间单独记录，批量耗时除以图片数注明为 batch 平均 ms/img。

4. 将 ViT P4/P8 的验证对照保存为 `patch_results.csv`，结合第三周 `results.csv` 汇总 CNN BN on/off、ResNet residual on/off、ViT P4/P8 的验证指标与预算；冻结六个变体及各自 best，准备一次统一测试。

### 分步操作与自查

#### 第一步：用小混淆矩阵核对指标

约定行为真实类、列为预测类。单类 F1 为 `2*TP/(2*TP+FP+FN)`，Macro-F1 为所有类别 F1 的平均。

计算某个类别 c 时，暂时把其他类别都视为“非 c”：TP（真正例）是真实为 c 且预测为 c；FP（假正例）是真实非 c 却预测为 c；FN（假负例）是真实为 c 却预测成其他类。Macro 表示每个类别占相同权重，不按该类样本数加权。本项目为每个类别算一次 F1，再求平均。

```text
两类例子：[[2,1],
           [0,1]]
accuracy = 3/4 = 0.75
F1(class0) = 4/5 = 0.8
F1(class1) = 2/3
Macro-F1 ≈ 0.733333
```

上例第 0 类的 TP=2、FP=0、FN=1，所以 F1=`4/(4+0+1)=4/5`；第 1 类的 TP=1、FP=1、FN=0，所以 F1=`2/(2+1+0)=2/3`。accuracy 只看对角线正确数 `(2+1)/4`，Macro-F1 则为 `(4/5+2/3)/2=11/15`，两者回答的问题不同。

先让指标函数通过这个例子。正式 CIFAR 使用 10 类，某类分母为 0 按约定记 0，不因它暂未出现在某批就改变平均类别数。

用 `y_true=[0,0,0,1]`、`y_pred=[0,0,1,1]` 实际构建这张矩阵：对每对类别执行 `matrix[true,pred] += 1`。第 c 类的 TP 是对角元素，FN 是第 c 行和减 TP，FP 是第 c 列和减 TP。参数 `num_classes=2` 时得到上表结果；再设为 3，未出现的第 2 类 F1=0，Macro-F1 应为 `22/45≈0.488889`。这两次检查验证矩阵方向和固定类别数口径。

<a id="classification-metrics"></a>
#### 第二步：把新指标接到 evaluate.py 的完整数据循环

创建 `src/petvision/evaluation/` 和空的 `__init__.py`，再创建 `src/petvision/evaluation/classification.py`，实现 `metrics_from_confusion(matrix)`：输入整份数据的整数混淆矩阵，按第一步的公式返回 accuracy 和 macro_f1。矩阵的类别数由 matrix.shape[0] 确定；它不负责加载数据或模型。先用上面的两类 / 三类例子检查。

在 evaluate.py 顶层导入这个函数。今天将 main 中原先仅调用 evaluate 获取汇总指标的部分，改为本脚本顶层的 `evaluate_classification(model, loader, criterion, device, num_classes)`；让它在同一次完整遍历中统计 loss、正确数、样本数和混淆矩阵，再返回包含这些结果的字典。训练器里的 evaluate 保持原接口。加载冻结模型和所选 loader、切 eval 后，在新函数循环外建立 `matrix = torch.zeros((num_classes,num_classes), dtype=torch.long)`；类别数来自 cfg.model.num_classes，CIFAR 为 10。循环内关闭梯度，取得 logits 和 predicted，再把标签和预测转到 CPU。按下方方式把当前批次累计到同一张矩阵：

```python
for true_id, pred_id in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
    matrix[true_id, pred_id] += 1
```

loss 继续按第二周的样本加权方法累计。循环结束后才调用 metrics_from_confusion，断言 matrix.sum() 等于实际 n_samples，并核对矩阵算出的 accuracy 与总正确数 / 总样本数一致；不能对每批 F1 取平均。

结果字典中加入 macro_f1、混淆矩阵的 `matrix.tolist()`、参数量 `sum(p.numel() for p in model.parameters())`。所有模型使用同一确定性预处理和同一数据划分，读取所选 config 的 experiment_dir，按 `f"{args.split}_metrics.json"` 写入对应验证或测试结果；保存方法复用第二周 JSON 输出。

metrics_from_confusion 用整数矩阵累计、用浮点除法计算 F1；总样本数为 0 时明确报错。两类 / 三类手算检查放在本地草稿或当前评估模块的检查入口，实际调用该函数，不仅在纸上验证公式。误分类采集仍可沿用第二周的独立函数，或在这次完整循环中同步收集；若同步收集，收满图片后仍遍历全部样本。

<a id="inference-timing"></a>
#### 第三步：区分加载耗时与推理耗时

加载 checkpoint 记录为单独阶段。推理先用固定 batch 预热，再重复测量；GPU 前后同步，记录 batch、dtype、device 和重复次数。

批量总时长除以处理图片数是平均 ms/img，不等于逐张交互请求的完整延迟。不同指标名称要反映实际测法。

先固定一个已搬到模型设备的图片批次，模型 eval；加载和数据读取均在下面片段之前完成。起始约定为预热 5 次、计时 20 次，所有模型使用相同设置：

```python
import time

with torch.no_grad():
    for _ in range(5):
        model(images)
    if images.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(20):
        model(images)
    if images.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
ms_per_image = seconds * 1000 / (20 * images.shape[0])
print("batch mean ms/img:", ms_per_image)
```

这是模型前向的批量平均值，验证指标仍由完整验证集单独计算。每条测时记录写 warmup=5、repeats=20、实际 batch、dtype、device 和是否包含数据搬运；本片段不包含搬运。如记录峰值，CUDA allocated 包含已加载模型等当前分配，不能写成整张显卡的总使用量。

将计时片段封装为 evaluate.py 顶层的 `benchmark_forward(model, images)`，返回包含 ms_per_image、warmup、repeats、batch_size、dtype、device、include_transfer=False 的字典。main 在完整评估后取所选 loader 的一批图片、迁移到设备，调用该函数，将返回值放入结果的 inference 字段，再写 metrics JSON。加载阶段的秒数另记 load_seconds，两个数字不能混用；六个模型都读取相同 batch 设置，并记录实际批大小。

#### 第四步：冻结六个变体

先把两个 ViT 的真实验证结果写入 `experiments/vit_patch_size/week5/patch_results.csv`，每行含 variant、split=val、token_count、best_epoch、val_loss、val_accuracy、macro_f1、params、epochs、batch、训练秒数、峰值显存（CUDA 可用时）、推理测量条件、config、checkpoint 和指标来源路径。指标读取各运行目录的 val_metrics.json，训练量与耗时读取各自真实记录；用第一周 CSV 写法保存两行，不能填入示例值。

再读取第三周的 `experiments/cnn_vs_resnet/week3/results.csv`，核对其中四组配置和 best；若第三周尚未计算 macro_f1，可用已冻结 best 和同一验证集补算，仍不访问测试集。

核对四个第三周模型与两个 ViT 的 config/best 路径，将 variant、config、checkpoint、验证指标和 best epoch 列为本周 README 的冻结清单。每一行直接指向实际运行的文件；冻结后按这张清单逐一测试。此时选择依据仍是验证集，准备 Day 5 统一测试。

评估输出保存在各自配置的 experiment_dir，文件名为 `val_metrics.json` 或 `test_metrics.json`，图表也按对应运行目录保存，避免六个变体互相覆盖。第五周的总表另外汇总这些结果，保留每行来源路径。

### 运行与验收

先用本节两类例子核对指标，再评估两个 ViT 的验证集：

```bash
python scripts/evaluate.py --config configs/cv/vit_cifar10.yaml --checkpoint outputs/vit_patch4/best.pt --split val
python scripts/evaluate.py --config configs/cv/vit_patch8_cifar10.yaml --checkpoint outputs/vit_patch8/best.pt --split val
```

每组应覆盖 `5000` 条样本，混淆矩阵元素总和也应为 5000。把实际指标与测时条件整理进 `patch_results.csv`，结合第三周验证结果冻结六份配置；今天不查看测试集指标。

- [ ] F1 和混淆矩阵经小例子核对，不能把 batch accuracy 直接无权平均。
- [ ] 测量条件清楚，跨架构差异与预算限制有记录。
- [ ] 所有配置和 best 在查看测试结果前已冻结。

<a id="day-5"></a>
## Day 5：统一 CIFAR-10 测试与错误分析

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [总计划验收标准](StudyPlan.md#acceptance) | 测试使用与结果模板 | 保存冻结后的测试结果 |
| [ViT 原论文](https://arxiv.org/abs/2010.11929) | 数据规模与归纳偏置讨论 | 限定小数据实验结论 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 依次评估冻结的六个 checkpoint |
| 1.5h | 保存混淆矩阵与错误案例 |
| 1h | 写实际比较结论与局限 |
| 0.5h | 整理图表和结果索引 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 使用独立 `evaluate.py` 对六个 checkpoint 评估官方 CIFAR-10 测试集，将实际结果写入 `visual_test_results.csv`，每行保留 config、checkpoint、split=test 与版本。

2. 保存主要 CNN/ResNet/ViT 的混淆矩阵和误分类样例，观察哪些类别容易混淆；结果差也完整保留。

3. 解释 BN、残差和 Patch 三类内部对照的实际结果；跨架构比较记录训练预算和参数量，避免把单次结果推广到所有视觉任务。

4. 测试后若发现实现错误，先修复并注明该轮结果无效；新的调参只使用训练/验证。CNN 70%+ 是参考目标，不以反复测试作为达标手段。

### 分步操作与自查

#### 第一步：逐个加载冻结 checkpoint

按下方六个测试命令运行，每次记录配置标识、checkpoint、split=test、样本数和指标。CIFAR-10 官方测试集有 10000 条，完整评估不得遗漏尾批。

若某个 checkpoint 丢失，明确标记该行未评估，先按原配置补齐训练；不要用另一个变体的权重填入它的行。

#### 第二步：结果表保留比较条件

除 accuracy/F1，记录参数量、epoch、batch、训练时间、推理测法和初始化来源。CNN/ResNet/ViT 均为从零训练，预训练模型结果以后另列。

将六个变体 experiment_dir 下的 test_metrics.json 整理成六行，用 csv.DictWriter 保存到 `experiments/vit_patch_size/week5/visual_test_results.csv`；至少包含 variant、split、n_samples、loss、accuracy、macro_f1、params、config、checkpoint、指标来源路径和实际库/代码版本。写入前确认每行 variant 与 Day 4 冻结清单一一对应，不能用验证指标填充缺失的测试结果。

#### 第三步：用图像辅助解释

每个主要模型查看少量误分类，关注真实类、预测类和相似类别。把样例和混淆矩阵联系起来，但不以挑选的几张图代替全部统计。

绘图也使用评估循环累计的同一张 matrix，classes 按基础 CIFAR10 dataset.classes 的原始顺序取得，不能临时按预测排序。在 evaluate.py 中，先创建 `fig, ax = plt.subplots()`，再用 `ax.imshow(matrix.numpy(), cmap="Blues")` 显示计数，添加 colorbar，横轴标预测类别、纵轴标真实类别。用 `ax.set_xticks(range(10), labels=classes, rotation=45, ha="right")` 和 `ax.set_yticks(range(10), labels=classes)` 标出实际类别，保存到当前 experiment_dir 的 `figures/confusion_test.png`，随后关闭图；验证图则用 confusion_val.png。imshow 的用法可查 [Matplotlib 官方接口](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html)。

矩阵按真实类为行、预测类为列绘图，坐标轴写实际 classes；对角线之和应等于正确数，全矩阵和应为 10000。误分类沿用第二周的采集方法，按 CIFAR 的三通道逆归一化并转 HWC 显示，保存为当前 `experiment_dir/figures/mistakes_test.png`，验证样例保存为 mistakes_val.png。总表每行链接该变体 experiment_dir 下的 test_metrics 和图表，避免不同 split 的图互相覆盖。matrix 若已转为结果中的列表，画图前用 torch.tensor 恢复；获取 classes 时，Subset 的基础数据集是 loader.dataset.dataset，官方测试集则直接是 loader.dataset。

#### 第四步：结论分成内部对照与跨模型比较

BN、残差、Patch 各自比较控制条件内的差异；跨架构表同时报告不同优化设置和计算预算。一次小数据实验不能证明某类模型在所有任务上更好。

### 运行与验收

```bash
python scripts/evaluate.py --config configs/cv/cnn_cifar10.yaml --checkpoint outputs/cnn_bn_on/best.pt --split test
python scripts/evaluate.py --config configs/cv/cnn_no_bn_cifar10.yaml --checkpoint outputs/cnn_bn_off/best.pt --split test
python scripts/evaluate.py --config configs/cv/resnet_cifar10.yaml --checkpoint outputs/resnet_on/best.pt --split test
python scripts/evaluate.py --config configs/cv/resnet_no_residual_cifar10.yaml --checkpoint outputs/resnet_off/best.pt --split test
python scripts/evaluate.py --config configs/cv/vit_cifar10.yaml --checkpoint outputs/vit_patch4/best.pt --split test
python scripts/evaluate.py --config configs/cv/vit_patch8_cifar10.yaml --checkpoint outputs/vit_patch8/best.pt --split test
```

每行都应为 `split=test`、`n_samples=10000`，混淆矩阵总数也应一致。汇总到 `experiments/vit_patch_size/week5/visual_test_results.csv`，保存主要模型的混淆矩阵与错误图。

- [ ] 六组测试结果来自冻结配置，样本数和预处理一致。
- [ ] 最终表区分验证与测试，失败结果没有被删除。
- [ ] 能联系实际样例解释卷积与自注意力的差异。

<a id="day-6"></a>
## Day 6：视觉报告与核心流程重写

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [第五周实验记录](../experiments/vit_patch_size/week5/README.md) | 本周真实配置与结果 | 完成视觉阶段总结 |
| [ViT 原论文](https://arxiv.org/abs/2010.11929) | 按自己的模型回顾 Patch→Encoder→CLS | 解释结构 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 核对结果表、曲线与复现路径 |
| 1.5h | 完成视觉报告与失败分析 |
| 1h | 有限参考重写 Patch/分类核心 |
| 0.5h | 更新每周 README 与环境记录 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 在本周 README 汇总 CNN、ResNet、ViT 的真实测试表、三类对照结论和错误案例，链接第三周结果与固定划分。

2. 有限参考下重写 Patch 投影、CLS 拼接、位置嵌入与 CLS 分类的核心片段，画出 P4/P8 全流程形状。

3. 确认模型能独立导入和加载，不把可复用 Encoder 写死为某个 T；记录实际版本、参数量、训练/推理时间。

4. 将预训练模型迁移、Oxford Pet、更多种子和调参列入拓展，本周按既定两个 Patch 完成主线。

### 分步操作与自查

#### 第一步：报告按实验问题组织

打开本周 README，在周总结中按以下顺序写四段，不需要重新复制全部训练日志：

1. BN 对照：从第三周结果表取 CNN on/off 两行，写明保持相同的条件、实际指标差异，并链接 BN 曲线。
2. 残差对照：同样取 ResNet on/off 两行，补充 projection 与参与训练参数的差异；不能用 CNN 对比 ResNet 来代替这组实验。
3. Patch 对照：引用本周 patch_results.csv、曲线和实测耗时，说明 P4/P8 的 token 数、参数量与结果如何变化。
4. 跨模型比较：引用冻结后的 visual_test_results.csv，列出三类模型的实际优化设置与预算，再描述本次观察；这段与前三组内部对照分开。
5. 每段末尾补一句局限或仍不确定的原因。写完逐个打开引用文件，确认数值来自该文件、split 正确，验证结果没有被写成测试结果。

#### 第二步：有限参考重写 ViT 核心

在草稿中写 Patch reshape/投影、CLS 扩展、位置相加、Encoder 和 CLS 分类。先写 P4/P8 的每个 shape，再核对实现。

可以复用已经验证的 Encoder，本次重写重点是视觉 token 到分类的连接，不需要重新训练一套模型。

用草稿的重写路径实际处理 B=1、B=2 的随机图片，分别检查 P4/P8 的 token 长度 65/17、CLS 特征 `[B,128]` 和 logits `[B,10]`，再做一次有限 loss/梯度检查。下方已有 test_vit 只检查它所导入的实现，草稿也要被实际调用。

#### 第三步：核对可复现材料

1. 以 Day 4 的六行冻结清单为起点，逐行找到 config、best、test_metrics 和图表；权重位于忽略目录时仍写清本地路径。
2. 对照 config 与 checkpoint，检查模型开关、Patch 大小和分类数匹配，再核对固定划分文件和预处理。
3. 将每行测试样本数与混淆矩阵总和核对为 10000，将矩阵对角和除以 10000，与该行 accuracy 比较。
4. 检查推理时间旁是否写明设备、dtype、batch、预热 / 重复次数和是否含搬运；缺测就标缺测，不能用其他模型的测量值补齐。
5. 在 README 保留实际命令和环境版本，逐个验证相对链接。补文档只读取已有产物，不为重新排版再次训练或评估测试集。

#### 第四步：记录真实局限

在 README 写单种子、训练预算、从零训练、小图像与模型规模的限制；把更多种子、预训练迁移和 Pet 数据集列为后续项目。

### 运行与验收

核心流程重写后，用小张量重新运行已有 ViT 检查：

```bash
python -m pytest tests/test_vit.py -q
```

核对六个最终结果、两组 Patch 曲线和来源配置的链接。今天整理已有结果与解释，已通过检查的大训练保留原记录即可。

- [ ] 视觉报告包含真实指标、配置、控制条件、局限和复现命令。
- [ ] 三类对照齐全，独立测试使用规则清楚。
- [ ] 有限参考能解释并重写 ViT 核心流程。

## 本周结束检查

- [ ] 小 ViT 复用手写 Encoder，Patch/CLS/位置形状与梯度检查通过。
- [ ] Patch=4/8 两组完成约定预算，并保存验证对照。
- [ ] CNN、ResNet、ViT 六个变体有冻结配置后的独立测试与失败分析。
- [ ] 结果、时间、参数量与跨架构局限集中到本周 README。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第六周](week6.md)。

