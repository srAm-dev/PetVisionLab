# 第三周逐日学习指南：CNN、ResNet 与 BN / 残差对照

> 更新日期：2026-10-31<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-1)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：手写 CNN、BasicBlock 和采用《动手学深度学习》结构的 ResNet18，在 CIFAR-10 上完成 BN on/off 与同深宽残差 on/off 两类对照，保留固定划分供第五周复用。<br>
> 环境：本地 Conda 环境 `petvision` 与实际 RTX 4060；沿用第二周 torch / torchvision、PyYAML、pytest，先以 batch=64、FP32 检查。

## 怎么使用这份指南

先确认[上一周指南](week2.md)的基础关卡已通过。应已跑通 MNIST 训练、验证和保存加载；本周复用训练器，新增数据和模型，不重写一套无关训练框架。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/cnn_vs_resnet/week3/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── src/petvision/
│   ├── data/datasets.py                    # Day 1：新增 CIFAR-10 与独立预处理
│   ├── models/cv/cnn.py                    # Day 2：CNN、BN 开关
│   ├── models/cv/resnet.py                 # Day 3–4：标准残差块与网络；Day 5：残差开关
│   └── utils/history.py                    # Day 2：保存训练记录与绘制曲线
├── configs/cv/
│   ├── cnn_cifar10.yaml                    # Day 2：BN on
│   ├── cnn_no_bn_cifar10.yaml              # Day 5：BN off
│   ├── resnet_cifar10.yaml                 # Day 4：残差 on
│   └── resnet_no_residual_cifar10.yaml     # Day 5：残差 off
├── scripts/inspect_cifar10.py              # Day 1：数据、划分与形状
├── scripts/train_cv.py                     # Day 2–5：组织三种运行模式与结果保存
├── scripts/evaluate.py                     # Day 2：扩展 CNN 评估，保留权重与划分核对
├── tests/test_resnet.py                    # Day 2–5：形状、反向、参考对照与残差开关
├── experiments/cnn_vs_resnet/week3/
│   ├── README.md                          # 已提供
│   ├── cifar10_split.json                 # Day 1：45k/5k 固定索引
│   ├── results.csv                        # Day 6：四组验证结果
│   ├── runs/<variant>/                    # Day 2–5：config、history、run_summary 与单组图
│   ├── configs/                           # Day 5：实际配置快照
│   └── figures/                           # Day 5–6：两类对照曲线
└── outputs/{cnn_bn_on,cnn_bn_off,resnet_on,resnet_off}/
    └── {initial,best,last}.pt             # initial 为训练前状态，best/last 为训练结果
```

## 开始前：激活环境并检查已有模块

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

解释器应位于 Conda 的 `petvision` 环境，`petvision.__file__` 应指向当前仓库的 `src/petvision/`。若出现 `No module named petvision`，先确认第二周已建立 `pyproject.toml` 和包目录，再在根目录执行 `python -m pip install -e . --no-deps`；若缺少 torch 或 CUDA 不可用，按[第二周 Day 1 的安装与排错步骤](week2.md#environment)逐项核对。

复用第二周的分类训练器，新增 CIFAR-10 数据与视觉模型。 本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

CIFAR-10 官方 50,000 条训练数据固定划分 45,000/5,000，种子 42，索引保存到 `experiments/cnn_vs_resnet/week3/cifar10_split.json`，第五周原样复用。从本周开始，新增的分类图像预处理统一使用 `torchvision.transforms.v2`，已有 MNIST 预处理保留原计算；第八周预训练 CLIP/VLM 使用各自配套的 processor。CIFAR 训练顺序为 ToImage → RandomCrop(32,padding=4) → RandomHorizontalFlip → ToDtype(torch.float32,scale=True) → Normalize，固定 mean/std=(0.5,0.5,0.5)；验证和测试只用 ToImage → ToDtype → Normalize。分别创建训练/验证 dataset 实例再套固定索引，避免共享随机增强。

Day 1–2 的 CNN 使用 32×32 输入。Day 4 起，ResNet 在上述随机增强之后、ToDtype 之前增加 Resize(96,96)，验证和测试也使用相同的确定性 Resize；具体操作见[教材结构与 96×96 输入](#d2l-resnet-input)。原始 CIFAR 图片和固定划分共用；data.image_size 记录预处理前的数据集图像尺寸，本例为 32。ResNet 的 96×96 是预处理 Resize 后实际送入模型的尺寸，两者分别说明。

CNN 起步为三层 3×3 Conv，通道 32/64/128，前两层后各做 2×2 Pool，最后 GAP→Linear(128,10)；BN 开关只替换归一化，同一对中卷积 bias 均保持 True。ResNet18 按教材使用 7×7,stride=2,padding=3 的入口卷积、BN、ReLU 和 3×3,stride=2,padding=1 的最大池化，接 [2,2,2,2] 个 BasicBlock、通道 64/128/256/512；卷积 bias=True，投影只用 1×1 Conv。保留 CIFAR 的 RGB 三通道，输入缩放到 96×96。四组从零训练，起始 batch=64、AdamW lr=1e-3、weight_decay=1e-4、20 epochs、FP32；先短运行测时，再冻结两组对照的预算。显存不足可统一减小同一对的 batch，正式结果需注明预算变化。测试集保留到第五周。

BN 两组内部都使用 32×32，残差两组内部都使用 96×96。第五周的 ViT 使用 32×32；跨架构表必须注明 image_size，结果反映各自完整配置的表现，不能只归因于架构。

<a id="cnn-terms"></a>
本周缩写先按下面理解，具体实现放在各天的步骤中：

| 名称 | 本项目中的具体含义 |
| --- | --- |
| CNN / Conv | 卷积神经网络 / 卷积层；沿图片空间滑动窗口提取特征 |
| BN | Batch Normalization，批归一化；本周用 BatchNorm2d 按通道处理卷积输出 |
| Pool / GAP | 局部池化 / 全局平均池化；GAP 将每个通道的全部 H×W 个位置平均成一个数 |
| BasicBlock / residual / shortcut | 基本残差块 / 残差连接 / 旁路；主分支算 F(x)，旁路提供与之相加的输入或投影结果 |
| stem / stage / depth | 网络入口部分 / 同一阶段的一组块 / 深度；[2,2,2,2] 表示四个 stage 各有两个块 |
| on / off / variant | 启用 / 关闭 / 实验变体；例如 cnn_bn_off 是关闭 BN 的那一组 |
| FP32 / lr | 32 位浮点数 float32 / learning rate 学习率；`1e-3=0.001` |

“冻结预算”指短检查后确定正式使用的 epoch、batch 等条件，并让同一对实验保持一致，不是冻结模型参数停止学习。“对齐”指按步骤把结构或初始参数逐项对应并核对数值，不能只因为设置相同 seed 就视为完成。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | 卷积形状与 CIFAR-10 固定划分 | 4h | 1h | 45k/5k 索引与预处理检查 |
| Day 2 | CNN、BatchNorm 与基线训练 | 4h | 1h | CNN 前反向与训练曲线 |
| Day 3 | 手写 BasicBlock 与投影分支 | 4h | 1h | 始终执行残差相加的普通块与投影块 |
| Day 4 | 按教材组装 ResNet18 并检查整网 | 4h | 1h | 96×96 RGB 输入、整网前反向与阶段形状检查 |
| Day 5 | BN 与残差的单变量对照 | 4h | 1h | 残差开关、四组训练、配置快照与验证记录 |
| Day 6 | 验证结果、错误分析与周总结 | 4h | 1h | 两类对照报告与第五周复用材料 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：卷积形状与 CIFAR-10 固定划分

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [CIFAR-10 教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/cifar10_tutorial.html) | 加载 CIFAR-10、卷积网络输入与类别 | 查看图片与一个批次 |
| [Transforms v2（官方英文）](https://docs.pytorch.org/vision/stable/transforms.html) | v2 推荐用法、图像类型与像素范围 | 按本页第二步建立训练 / 验证预处理 |
| [Conv2d 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Conv2d.html) | kernel、stride、padding 与输出形状 | 手算一次卷积尺寸 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读卷积、步幅、填充与池化 |
| 2h | 扩展数据加载和保存固定划分 |
| 1h | 检查增强隔离、图片和形状 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 用输出尺寸公式 `floor((H+2P-dilation*(K-1)-1)/stride+1)` 手算 32×32 经 3×3、padding=1 的 stride=1/2 后分别得到 32/16；池化同样核对尺寸。

2. 在 `src/petvision/data/datasets.py` 增加 CIFAR-10 构建函数，下载到 `datasets/raw/`，生成种子 42 的 45,000/5,000 索引并保存到本周实验目录；存在索引时直接读取。

3. 创建 `scripts/inspect_cifar10.py`，检查训练/验证索引不相交、类别映射一致。训练和验证实例分别持有各自 transform；重复取同一验证样本应一致。

4. 打印输入 `[B,3,32,32]` 和标签 `[B]`，查看增强前后少量图像。官方测试集只保留来源记录，本周不报告测试指标。

### 分步操作与自查

#### 第一步：先手算空间尺寸

本步做一个小练习：先计算图片经过卷积和池化后的形状，再用 PyTorch 检查。暂时使用随机图片，不需要下载 CIFAR-10 或训练模型。可新建临时脚本 `tmp/conv_shapes.py` 保存下方验证代码；后续的数据检查入口仍按第二步创建。

**1. 先读懂输入形状。**

CIFAR-10 的图片有 RGB 三个通道，高度和宽度都是 32 像素。一批图片的形状写作 `[B,C,H,W]`，这里就是 `[B,3,32,32]`：

| 符号 | 含义 | 本例取值 |
| --- | --- | --- |
| B | batch size，一批图片的张数 | 验证代码取 4 |
| C | channels，通道数 | 输入为 3，分别对应 RGB |
| H | height，图片或特征图的高度 | 输入高度为 32 像素 |
| W | width，图片或特征图的宽度 | 输入宽度为 32 像素 |

注意：形状中的数字要按位置理解。`[B,32,32,32]` 的第一个 32 是通道数，后两个 32 才是高度和宽度。

**2. 理解 H=32、K=3、P=1、S=1 分别是什么意思。**

下面先计算高度方向，假设输入高宽相同，卷积核为正方形，高宽方向的 padding、stride 也相同：

| 符号 | 含义 | 对应的 PyTorch 参数 | 本例的具体意思 |
| --- | --- | --- | --- |
| H | 输入高度 | 来自输入 Tensor 的形状 | H=32：输入有 32 行像素 |
| K | 卷积核的空间边长 | `kernel_size=3` | K=3：卷积窗口覆盖 3×3 的空间区域 |
| P | 每一侧的填充宽度 | `padding=1` | P=1：上下左右各补 1 圈零；高度从 32 临时补到 34，宽度也补到 34 |
| S | 窗口移动的步幅 | `stride=1` | S=1：窗口每次沿当前方向移动 1 格；S=2 则每次移动 2 格 |

本步使用默认的 `dilation=1`，表示卷积核的相邻采样点之间不留间隔。输出高度的公式为：

```text
H_out = floor((H + 2×P - K) / S) + 1
```

`floor` 表示向下取整，例如 `floor(15.5)=15`。`2×P` 是因为高度方向上方和下方各填充 P 行；宽度方向同理。公式可以理解为：填充后的高度减去窗口高度，得到窗口还能移动的距离；除以步幅并向下取整，得到可移动的次数；再加上初始位置的一次，就得到输出位置数。

把 **H=32、K=3、P=1、S=1** 代入：

```text
H_out = floor((32 + 2×1 - 3) / 1) + 1
      = floor(31 / 1) + 1
      = 32
```

输入宽度 W 也是 32，所以把公式中的 H 换成 W，得到 `W_out=32`。若仅把步幅改为 **S=2**：

```text
H_out = floor((32 + 2×1 - 3) / 2) + 1
      = floor(15.5) + 1
      = 16
```

宽度同样变为 16。这里的公式只计算空间尺寸，不计算输出通道数；输出通道数由 `out_channels` 单独指定。

**3. 把参数和完整形状对应起来，再计算池化。**

`nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)` 接收 3 通道输入，输出 32 个特征通道；根据刚才的计算，高宽保持为 32。这里的 32 个通道不是 32 个类别，最终类别分数由后面的分类头产生。

在它后面接 `nn.MaxPool2d(kernel_size=2, stride=2)`，池化会在每个通道内用 2×2 窗口取最大值，默认 padding=0、ceil_mode=False。高度计算为 `floor((32-2)/2)+1=16`，宽度同样为 16，通道数保持不变：

```text
输入                                  [B,3,32,32]
Conv2d(3→32, K=3, P=1, S=1)           [B,32,32,32]
MaxPool2d(K=2, S=2, P=0)              [B,32,16,16]
```

输出尺寸与**感受野**是两个问题：前者问“还有多少个位置”，后者问“一个位置能受到原图多大范围的影响”。这里卷积的一个输出位置对应原图 3×3 区域；池化合并相邻 2×2 个卷积位置后，对应区域为 4×4，而不是 6×6，因为相邻卷积窗口有重叠。

需要逐层手算时，记 `r` 为理论感受野边长，`j` 为相邻输出位置在原图上的间隔，从 `r=1,j=1` 开始。普通卷积 / 池化每层按 `r_new=r+(K-1)*j`、`j_new=j*S` 更新（这里 dilation=1；右边都用更新前的值）。上述卷积后 `r=3,j=1`，池化后 `r=4,j=2`。这是卷积 / 池化路径的几何覆盖范围；边缘可能包含 padding，实际像素贡献也未必相同。

**4. 用随机输入验证手算结果。**

把下面代码写进 `tmp/conv_shapes.py`：

```python
import torch
from torch import nn

images = torch.randn(4, 3, 32, 32)
conv_s1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
conv_s2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

with torch.no_grad():
    output_s1 = conv_s1(images)
    output_s2 = conv_s2(images)  # 分别作用于原始图片，不串联两个卷积。
    pooled = pool(output_s1)   # 池化接在 stride=1 的卷积后面。

print("输入：", images.shape)
print("stride=1 卷积：", output_s1.shape)
print("stride=2 卷积：", output_s2.shape)
print("stride=1 卷积后池化：", pooled.shape)

assert output_s1.shape == (4, 32, 32, 32)
assert output_s2.shape == (4, 32, 16, 16)
assert pooled.shape == (4, 32, 16, 16)
```

在项目根目录运行 `python tmp/conv_shapes.py`。输出与手算一致、三个断言通过，并能解释 H/K/P/S 及通道数的含义，就完成了这一步。随后再做 CIFAR-10 下载与划分。

<a id="cifar-data"></a>
<a id="cifar-v2-transforms"></a>
#### 第二步：统一加载接口，再准备 CIFAR-10 与预处理

<a id="unified-loader-interface"></a>
先把 MNIST 的预处理从加载函数中移到调用者，再按同一接口实现 CIFAR-10。按下面顺序修改加载函数、迁移调用处并执行检查；若已有实现，逐项核对即可。完成情况和实际检查结果记录在[第三周 README](../experiments/cnn_vs_resnet/week3/README.md#progress)。

**1. 先确定两个函数共同接收什么、返回什么。**

`build_mnist_loaders` 与 `build_cifar10_loaders` 使用相同的参数名称和顺序：

```text
data_dir, split_path, batch_size, seed, train_transform, eval_transform,
validation_size=5000, num_workers=0
```

前四项继续指定数据位置、划分文件、批大小和种子。train_transform 只处理训练图片；eval_transform 同时处理验证和测试图片，应采用确定性操作。因此统一叫 eval_transform，原 CIFAR 草稿中的 val_transform 也改为这个名字。两个函数都按固定顺序返回 `train_loader, val_loader, test_loader`，迭代得到 `(images, labels)`；类别通道与图像大小仍由实际数据集决定。

**2. 把 MNIST 的预处理创建移到调用前。**

打开 `src/petvision/data/datasets.py`，原先 build_mnist_loaders 接收 mean/std 并在内部创建 Compose，现在改为接收两个已经创建好的 transform。加载函数只负责读取数据、管理划分和创建 DataLoader。mean/std 仍保留在 YAML 的 data.normalize 中，由调用脚本读取并交给 Normalize，不能从配置中删掉。

MNIST 沿用原来的 `transforms.ToTensor()` → `transforms.Normalize(...)`。训练脚本与评估脚本从配置取 mean/std，两个独立检查脚本使用原来的 `(0.5,)`。先创建 eval_transform；当前 MNIST 没有随机增强，可以让 train_transform 指向同一个确定性 Compose，再通过两个具名参数传入加载函数。这里统一的是加载接口，MNIST 本次不同时迁移到 v2，以便核对旧权重与预处理保持一致；新增 CIFAR 路径按下文使用 v2。

**3. 为训练和验证分别创建 Dataset，再套各自索引。**

在 MNIST 加载函数中依次创建 train_base、val_base、test_set：前两者都读取官方训练集，但分别绑定 train_transform 和 eval_transform；test_set 读取官方测试集并绑定 eval_transform。随后让 train_set 使用 `Subset(train_base, train_indices)`，val_set 使用 `Subset(val_base, val_indices)`。

关键是两个 Subset 的底层 Dataset 必须不同。不能继续让验证引用 train_base，也不能创建 Subset 后修改共享底层对象的 transform，否则训练增强可能进入验证。两个 Dataset 读取同一份本地训练文件，不代表重新划分两套训练数据；仍使用原先保存的索引。

**4. 逐个迁移四处实际调用，不只修改函数定义。**

| 调用文件 | 在调用前做什么 | 传入什么 |
| --- | --- | --- |
| `scripts/train_cv.py` | 读配置中的 normalize，创建 MNIST 确定性 Compose | train_transform 与 eval_transform；当前两者可相同 |
| `scripts/evaluate.py` | 完成配置核对后，按相同 normalize 创建评估 Compose | 两个位置都传 eval_transform；只使用返回的 val/test loader |
| `scripts/inspect_mnist.py` | 创建 ToTensor 与 mean/std=(0.5,) 的 Normalize | 两个位置都传该确定性 Compose，继续检查与画图 |
| `scripts/check_mnist_training.py` | 创建同样的确定性 Compose | 两个位置都传该 Compose，保证固定 64 样本检查不引入随机增强 |

各脚本顶层导入 `from torchvision import transforms`，main 中先创建变换，再调用加载函数。删除调用中的旧 `mean=...`、`std=...`，其余路径、seed、validation_size、num_workers 沿用原值。旧签名不再保留兼容分支：漏传两个 transform 会报缺少参数，继续传 mean/std 会报不支持的关键字，应回到调用处修正。

**5. 检查接口迁移没有改变划分、批次和已有验证结果。**

在 `tests/test_datasets.py` 中编写接口检查，使用不下载数据的替身 Dataset，给训练与评估传入明显不同的变换，检查训练变换不影响验证 / 测试；再更换预处理、重新调用加载函数，检查原 split 文件、索引和取样顺序保持不变。同时核对图片 / 标签形状、dtype，以及已有划分与 validation_size 冲突时仍明确报错。原 `tests/test_cv_training.py` 继续检查模型、训练统计和 checkpoint。

从项目根目录激活 petvision 后，先执行：

```bash
python -m pytest tests/test_datasets.py tests/test_cv_training.py -q
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --smoke
python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split val
```

smoke 应只训练两批、验证两批，不写正式训练产物。独立验证应覆盖 5000 条，准确率与迁移前同一 checkpoint 的验证记录一致；loss 按实际设备容差比较。实际数值从第二周 README 和对应运行记录读取，不在计划中固定为某次实验结果。不要为了迁移接口重跑正式五轮训练或测试集评估。需要查看图片或检查固定小样本时，继续使用第二周的 inspect_mnist.py / check_mnist_training.py 命令。

**6. 按相同接口继续实现 CIFAR-10。**

CIFAR-10 是三通道 32×32 图片、10 类，官方训练集 50000 张、测试集 10000 张。在 datasets.py 中定义或继续完善 `build_cifar10_loaders`，完成本节的数据实例与下一步的固定划分 / DataLoader；接口采用上面的公共参数。两个 transform 由调用者传入，正式训练可以增强，过拟合检查改用确定性评估变换；validation_size 和 num_workers 也从配置传入。

先在 `scripts/inspect_cifar10.py` 的 main 中建立本日条件：data_dir 为 `datasets/raw`，split_path 为本周 `cifar10_split.json`，batch_size=64、seed=42。在顶层导入 torch 和 `from torchvision.transforms import v2`，main 中创建以下两个变换；下方导入语句仍放在文件顶层：

```python
import torch
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
eval_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

这条流程处理 CIFAR10 返回的 RGB PIL 图片，逐项含义如下：

| 操作 | 具体作用与输出 |
| --- | --- |
| `v2.Compose([...])` | 按列表顺序执行各个变换 |
| `v2.ToImage()` | 转为 `[3,32,32]` 的图像 Tensor（tv_tensors.Image 是 Tensor 的子类）；本题仍为 uint8、0–255，不缩放像素 |
| `v2.RandomCrop(32,padding=4)` | 上下左右各补 4 像素，从 40×40 中随机裁回 32×32；只用于训练 |
| `v2.RandomHorizontalFlip()` | 默认以 0.5 的概率左右翻转；只用于训练 |
| `v2.ToDtype(torch.float32,scale=True)` | 将本题 uint8 图像转为 float32，并从 0–255 缩放到 0–1；只转换 dtype 而省略 scale=True 不会完成这次数值缩放 |
| `v2.Normalize(mean,std)` | 按通道计算 `(x-mean)/std`；这里三个通道都使用 0.5，所以范围变为约 [-1,1] |

张量转换分成 ToImage 和 ToDtype 两步，详见 [ToImage](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToImage.html)、[ToDtype](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html)。不使用已弃用的 [v2.ToTensor](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html)。第二周已完成的 MNIST 实验保留当时的实现与记录；本周新增 CIFAR 路径使用上述 v2 流程，后续 ViT 也复用它。引用教程若仍展示旧版写法，数据来源与模型讲解仍可参考，预处理按本页实现。

先检查一张图片：ToImage 后 dtype 为 uint8，ToDtype 后为 float32 且值在 [0,1]，Normalize 后在 [-1,1]。训练增强发生在 uint8 阶段，Normalize 前再转浮点数；不要在已经缩放的图片上重复除以 255。使用原有 DataLoader 即可将图片组成 `[B,3,32,32]` 批次，无需为本练习自定义 collate_fn。

在数据函数内用传入的参数创建原始数据实例；首次下载通过 download=True 明确开启。训练 / 验证实例分别持有各自的 transform：

```python
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

train_base = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
val_base = CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
test_set = CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
```

先检查三个实例长度为 50000、50000、10000。train_base 与 val_base 读取相同训练图片，分别持有各自的 transform；它们还不是最后的 45k/5k 子集。类别名称取 `train_base.classes`，编号顺序与 targets 对应。

#### 第三步：保存 45k/5k 索引，再创建三个 DataLoader

复用[第二周 MNIST 的索引保存和批次步骤](week2.md#mnist-data)：本周只对 0..49999 生成排列，前 45000 个作为 train_indices，其余 5000 个作为 val_indices。JSON 的 dataset 改为 cifar10、n_total 改为 50000，保存到传入 split_path；存在文件时读取并核对，不因开始新模型就覆盖划分。

函数内部先校验 validation_size 为满足 `0 < validation_size < 50000` 的整数，再计算 `n_train = n_total - validation_size`，用 `indices[:n_train]` / `indices[n_train:]` 切分；已有文件的长度校验也使用这两个变量。本周正式配置固定 5000，默认得到 45000/5000。验证参数是否生效时另用临时 split_path 和 validation_size=10000，应得到 40000/10000；不要改写跨周复用的正式划分。

完成长度、范围、组内唯一性、交集和并集断言后，在数据函数内接上：

```python
train_set = Subset(train_base, train_indices)
val_set = Subset(val_base, val_indices)
```

随后按[第二周 DataLoader 示例](week2.md#mnist-loaders)分别构建三个 loader：训练 shuffle=True 并使用独立 Generator，验证 / 测试 shuffle=False，三者 num_workers 从参数读取、drop_last=False。函数最后返回三个 loader；不要把 test_set 放进训练 / 验证的随机划分。


#### 第四步：在检查脚本调用函数，验证增强隔离

在 inspect_cifar10.py 顶层导入 `build_cifar10_loaders`，在创建 transform 之后调用，传入上面约定的 data_dir、split_path、64、42 和两个 transform。打印三个 loader.dataset 的长度，取一个训练 batch 检查图片 / 标签；用入口保护调用 main。

对同一验证索引连续取两次，图片 Tensor 应相同。训练增强可能改变样本，但随机两次相同并不说明增强失效，查看 transform 定义并重复抽查。

画图前逆归一化；打印 labels 的最小/最大值，确认类别顺序来自同一数据集定义。

复用第二周 JSON 读写与集合断言，将 dataset 改为 `cifar10`、总数改为 50000，检查 45000/5000、组内无重复、交集为空、并集覆盖全部训练索引。用 `torch.equal` 比较同一验证图片的两次读取结果；三通道画图先逆变换 `image*0.5+0.5`，再 `permute(1,2,0)` 转为 HWC。额外打印一个 Conv 的 weight.shape，例如 `Conv2d(3,32,3)` 应为 `[32,3,3,3]`，分别对应输出通道、输入通道和核高宽。

### 运行与验收

```bash
python scripts/inspect_cifar10.py
```

应打印 `45000/5000` 的训练/验证数量，图片通常为 `[64,3,32,32]`，标签为 `[64]` 的 long Tensor；同一验证样本重复读取应一致。确认 `experiments/cnn_vs_resnet/week3/cifar10_split.json` 已保存，第二次运行读取原文件。

- [ ] 卷积输出尺寸手算与实际 Tensor 一致。
- [ ] 训练/验证索引和预处理独立，保存的划分可供第五周复用。
- [ ] 知道卷积 weight 的输出通道、输入通道、核尺寸含义。

<a id="day-2"></a>
## Day 2：CNN、BatchNorm 与基线训练

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [CIFAR-10 教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/cifar10_tutorial.html) | 定义网络、训练与预测 | 将 MLP 训练器接上 CNN |
| [BatchNorm2d 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.BatchNorm2d.html) | 训练/评估统计与通道维度 | 检查 train/eval 的不同 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读 CNN 与 BN，画出尺寸变化 |
| 2h | 实现 CNN，拆分训练入口，再接通配置和 smoke |
| 1h | 小批过拟合并启动正式基线 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `src/petvision/models/cv/cnn.py`，按本周结构实现可开关 BN 的 CNN；每个阶段打印一次形状，最终输出 `[B,10]`。

2. 创建 `configs/cv/cnn_cifar10.yaml`。先提取 `train_cv.py` 已有的三种运行模式，再在 main 中按顺序扩展数据、模型和优化器选择，接入 CIFAR。沿用第二周的按样本加权指标、best checkpoint、history 和种子处理。

3. 在 `tests/test_resnet.py` 添加 CNN 的 batch=1/4 前向与反向检查。检查 BN 的 running statistics 属于 buffer，评估需要 model.eval()。

4. 先执行 smoke 和固定 64 样本过拟合；记录一个 epoch 的实际耗时，冻结正式预算后重建 CNN BN on 和优化器，在首次训练前保存 `outputs/cnn_bn_on/initial.pt`，再运行基线并保留验证曲线和 best。训练等待时阅读后续残差材料，不重复计时。

### 分步操作与自查

#### 第一步：把 CNN 拆成可解释的阶段

创建 `src/petvision/models/cv/cnn.py`，顶层写 `import torch` 和 `from torch import nn`，在其中定义继承 nn.Module 的 CNN。沿用第二周的 models/cv 包目录和空 __init__.py；下方结构写入构造函数，forward 依次计算各阶段并返回 logits。

```text
Conv3→32 + BN? + ReLU + Pool    [B,32,16,16]
Conv32→64 + BN? + ReLU + Pool   [B,64,8,8]
Conv64→128 + BN? + ReLU         [B,128,8,8]
AdaptiveAvgPool(1)              [B,128,1,1]
Flatten + Linear128→10         [B,10]
```

其中 GAP 指全局平均池化，将每个通道的空间位置取平均；Pool 使用 `nn.MaxPool2d(2,2)`，GAP 使用 `nn.AdaptiveAvgPool2d(1)`，随后 `torch.flatten(x,1)` 保留 batch 维。类名约定为 `CNN`，构造参数接收 num_classes 和 use_bn；第三周训练入口按 model.name 选择它并传入配置字段。

`use_bn=False` 时以 Identity 替换 BN。两组 Conv 的 bias 都保持 True，确保这个设置不会随 BN 开关一起变化。

落实到第一段代码，就是 `Conv2d(3,32,3,padding=1)` → `BatchNorm2d(32)` 或 `Identity()` → ReLU → `MaxPool2d(2,2)`。三层卷积均用 kernel_size=3、stride=1、padding=1，保持各自输入的高宽；只有前两次池化缩小尺寸。第三段输出 `[B,128,8,8]`，GAP 把每个通道的 64 个数平均，得到 `[B,128,1,1]`，展平为 `[B,128]` 后再分类。

BN 输入为 `[B,C,H,W]` 时，每个通道用 B×H×W 个值统计均值和方差，输出形状不变。默认设置下 train 模式使用当前批次统计并更新 running_mean/running_var；eval 模式使用已积累的统计。gamma、beta 是每通道一对可学习的缩放和偏移参数；running statistics 是保存并随设备迁移的 buffer，不由优化器求梯度更新。Identity 原样返回输入，表示该位置不做 BN。

按本结构计算：BN off 参数量为 94538，BN on 多 448 个 gamma/beta 参数，为 94986；running mean/variance 属于 buffer，不计入 parameters。若实现结构变化，重新列明差异。

#### 第二步：创建首份 CIFAR 配置

创建 `configs/cv/cnn_cifar10.yaml`，沿用第二周字段层级。以下给出完整起始配置，训练入口需要读取这些字段：

```yaml
model:
  name: cnn
  num_classes: 10
  use_bn: true
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
  learning_rate: 0.001
  weight_decay: 0.0001
experiment_dir: experiments/cnn_vs_resnet/week3/runs/cnn_bn_on
output_dir: outputs/cnn_bn_on
```

`augmentation` 只用于训练 dataset，验证使用相同 normalize；过拟合检查关闭随机增强。`training.optimizer: adamw` 需要入口创建 AdamW，不能继续无条件使用第二周的 Adam。数据根目录、变换、学习率与输出路径都从当前配置读取，不能依靠上一次运行遗留的全局变量。

先保存这份配置，按下一步整理入口。数据、模型和优化器的创建先按顺序放在 `main()` 中；较长的运行模式和记录工具分别提取。`cv_trainer.py` 继续接收模型、loader、优化器、损失函数和设备，计算方法沿用第二周。

<a id="cv-entry-refactor"></a>
#### 第三步：先整理训练入口，再接入 CIFAR

第二周的 `main()` 同时包含配置读取、对象创建、smoke、过拟合、正式训练和保存结果。只有一种数据和模型时还能顺着读；新增 CNN、增强和资源统计后，应先按职责提取普通函数。下面每一小步完成后再继续，先用 MNIST 确认提取前后行为一致。

**3.1 先明确各文件负责什么。**

| 文件 | 负责的事情 | 接收与返回 |
| --- | --- | --- |
| `src/petvision/data/datasets.py` | 读取数据、管理固定划分、创建 DataLoader | 接收路径与已经创建好的 transform，返回三个 loader |
| `src/petvision/training/cv_trainer.py` | 执行一轮训练或一次评估，按样本统计指标 | 接收对象，返回 loss、accuracy、n_samples |
| `src/petvision/utils/history.py` | 将已有指标保存为 CSV 或绘制曲线 | 接收 history 或 row 与输出目录，写出文件 |
| `scripts/train_cv.py` | 解析命令，准备对象，选择运行模式，保存正式结果 | 接收命令行参数，打印检查结果或生成实验文件 |
| `scripts/evaluate.py` | 核对 checkpoint，重建模型与数据，独立评估 | 接收配置和权重路径，打印或保存指定 split 的结果 |

当前 MLP / CNN 和 Adam / AdamW 的构造都只有几行，直接在入口按名称选择即可。优先提取较长的运行流程，以及可以直接复用的训练计算、数据划分和记录工具。训练入口和独立评估中保留少量相同的构造代码；以后确实出现多处需要同步维护的复杂规则，再提取对应的共享函数。

**3.2 提取三种运行模式，再把记录与绘图工具移到 utils。**

在 `main()` 前定义下面三个顶层函数，函数体从已有分支移动过来。移动一段逻辑时，同时列出它原来读取的变量，再把这些变量写成函数参数；函数内建立自己的局部变量。

| 建议接口 | 从原 main 移入的内容 | 自查 |
| --- | --- | --- |
| `run_smoke(model, train_loader, val_loader, optimizer, criterion, device)` | 训练与验证各最多两个批次，打印指标 | 仍使用 islice；只打印检查结果 |
| `run_overfit(model, train_loader, optimizer, criterion, device, n_samples)` | 检查样本数量，固定一批图片，初始评估、200 次更新与最终评估 | 用 n_samples 替换原来的 args.overfit_samples；批次在循环外只取一次 |
| `run_training(cfg, model, train_loader, val_loader, optimizer, criterion, device)` | epoch 检查、split hash、配置保存、完整训练循环、best/last、history 和曲线 | best_accuracy、history、experiment_dir 都在函数内建立 |

提取后，`main()` 中原来的三个大分支各剩一次函数调用。先判断 smoke，调用后 `return`；再判断 `args.overfit_samples is not None`，调用后 `return`；最后调用 `run_training`。`return` 用来明确结束检查模式。

接着将保存与绘图函数提取到单独模块，按下面顺序操作：

1. 新建 `src/petvision/utils/history.py`，沿用已有 utils 包与 `__init__.py`。把原来的 `save_history(history, experiment_dir)` 和 `plot_history(history, experiment_dir)` 完整移入，先保持参数、CSV 列、文件名和绘图逻辑一致。
2. 将这两个函数需要的 `import csv`、`from pathlib import Path`、`import matplotlib.pyplot as plt` 放在 history.py 顶层。该模块只定义函数，保存和绘图在调用函数时执行。
3. 删除 train_cv.py 中原来的两个函数定义，顶层改为 `from petvision.utils.history import save_history, plot_history`。移走仅供这两个函数使用的 csv 和 pyplot 导入；train_cv.py 的 Path 仍用于命令行参数、split hash 和结果目录，继续保留。
4. `run_training` 中原来的两处调用保持相同参数。输出目录由调用者传入，函数只处理已经算好的指标；数据读取、训练计算和运行模式选择由对应模块负责。生成的 history.csv、figures/training_curves.png 继续写入配置的 experiment_dir，源码统一放在 src 下。

后面的第六步再向 history.py 添加 `append_history`；这三个函数放在同一个模块，便于复用。`run_smoke`、`run_overfit`、`run_training` 仍放在 train_cv.py 中，负责组织本入口的流程。

已有 checkpoint 检查中，`monkeypatch.setattr(train_cv, "plot_history", ...)` 仍然有效：train_cv 顶层导入了这个名称，run_training 也通过这个名称调用。此处继续保留测试的替换位置。

进入检查模式时，数据函数仍可读取或首次生成固定划分；实验配置快照、history、曲线和 checkpoint 由正式模式保存。

此时运行 MNIST smoke 和已有训练检查，确认提取函数没有改变计算和 checkpoint 行为。再运行固定 64 条 MNIST 样本的过拟合检查。先完成这一轮，再扩展数据名称。

**3.3 在 main 的数据准备部分扩展预处理。**

在 `train_cv.py` 顶层补 `from torchvision.transforms import v2`，保留 MNIST 使用的 `from torchvision import transforms`。main 读完配置后，删除第二周“仅支持 mlp / mnist / adam”的三个固定拒绝条件；后面的实际选择分支分别对未知名称报错。保留配置顶层为字典与 dtype=float32 的校验。

在种子和设备选择之后，先取 `data_cfg = cfg["data"]`，再读取 normalize，根据 data.name 创建 train_transform、eval_transform：

1. `name == "mnist"`：保留原来的 `transforms.ToTensor()` → `transforms.Normalize(...)`；当前没有随机增强，两个返回位置使用同一套确定性变换。
2. `name == "cifar10"`：先检查 `image_size == 32`。mean/std 列表各有三个数，分别对应 RGB 通道；读取 `augmentation.random_crop_padding` 和 `random_horizontal_flip`。
3. 创建 CIFAR 的 eval_transform：`v2.ToImage()` → `v2.ToDtype(torch.float32, scale=True)` → `v2.Normalize(...)`。它始终使用确定性预处理。
4. 训练增强启用时，先建立以 `v2.ToImage()` 开始的操作列表；padding 大于 0 时加入 `v2.RandomCrop(image_size, padding=padding)`，flip 为 true 时加入 `v2.RandomHorizontalFlip()`；最后加入 ToDtype 和 Normalize，再用 `v2.Compose` 包起来。padding 必须为非负整数，flip 必须为布尔值。padding=0 时省略裁剪；flip=true 使用默认概率 0.5。
5. 判断 `args.overfit_samples is not None`，如果正在做过拟合检查，就令 train_transform 使用 eval_transform。只覆盖本次使用的局部变量，保持 YAML 中正式训练的增强设置。正式训练和 smoke 使用配置指定的训练增强。
6. 未知数据名称明确抛出 ValueError。自查两个变量都是可调用的 transform，而不是已经变换过的图片。

这一步最终只准备好两套预处理，接着再创建 loader。未来新增数据集时，在对应数据分支中明确它的尺寸、通道、增强和归一化；同一个 CIFAR 数据分支可以继续用于 CNN、ResNet 和 ViT。

**3.4 选择已有数据函数，公共参数只写一次。**

在 `train_cv.py` 顶层导入 Day 1 的 build_mnist_loaders、build_cifar10_loaders。main 中先完成上一步的预处理，再根据 data.name 选择加载函数。因为 Day 1 已统一签名，MNIST 与 CIFAR 可以共用下面这段参数连接：

```python
if data_cfg["name"] == "mnist":
    loader_builder = build_mnist_loaders
elif data_cfg["name"] == "cifar10":
    loader_builder = build_cifar10_loaders
else:
    raise ValueError(f"不支持的数据集：{data_cfg['name']}")

train_loader, val_loader, _ = loader_builder(
    data_dir=data_cfg["root"],
    split_path=data_cfg["split_path"],
    batch_size=data_cfg["batch_size"],
    seed=cfg["training"]["seed"],
    train_transform=train_transform,
    eval_transform=eval_transform,
    num_workers=data_cfg["num_workers"],
    validation_size=data_cfg["validation_size"],
)
```

这段代码写在 main 中，替换原来只调用 build_mnist_loaders 的位置。`loader_builder` 保存的是选中的函数本身，最后加括号才执行它。训练入口使用返回的前两个 loader；第三个测试 loader 用 `_` 接收，本周训练不使用它。

自查：MNIST 返回 55000/5000/10000，CIFAR 返回 45000/5000/10000；两者使用各自配置中的 split_path。用独立检查命令查看实际 transform 和一个批次，验证图片重复读取应一致。main 准备对象时只创建 loader，把训练批次的读取留给所选运行模式，避免提前消耗训练顺序。

已有 `tests/test_cv_training.py` 中，`test_checkpoint_best_and_last` 仍可替换 `train_cv.build_mnist_loaders` 来提供小数据。main 选择该函数时会取到测试替身；验证分数替身和 best/last 断言继续保留，测试仍调用真实 main 并通过临时目录检查保存的文件。

**3.5 在 main 中直接扩展模型和优化器选择。**

在 `train_cv.py` 顶层导入 CNN，保留 MLP。main 的数据准备之后，取 `model_cfg = cfg["model"]`，用一组 if / elif / else 创建模型：

- `name == "mlp"`：向 MLP 传入 input_dim、hidden_dims、num_classes，保留第二周尺寸。
- `name == "cnn"`：创建 `CNN(num_classes=model_cfg["num_classes"], use_bn=model_cfg["use_bn"])`。
- 其他名称：抛出说明实际名称的 ValueError。此时先实现 MLP / CNN，Day 4 再添加 ResNet18。

选中模型后，统一执行 `model = model.to(device)`，然后创建 CrossEntropyLoss。

接着取 `training_cfg = cfg["training"]`，再用一组分支创建优化器：optimizer=adam 时创建 `torch.optim.Adam`，optimizer=adamw 时创建 `torch.optim.AdamW`，两者都从当前配置读取 learning_rate、weight_decay，参数来自已移动到设备的 model。未知名称明确报错。

正式训练要求 epochs 为正整数，这项检查留在 run_training 中。image_size=32 的检查留在 CIFAR 预处理分支，在读数据前完成。自查 YAML 改成 cnn / cifar10 / adamw 后，实际对象确实随配置变化。新增一种图像分类模型时，通常只增加模型构造分支，继续复用数据和训练循环。

**3.6 核对 main 的顺序和运行模式。**

按下面的顺序检查 main。对象创建按数据、模型、优化器分段排列，已经提取的三个运行函数负责后面的实际执行：

```text
解析 args，读取并检查 cfg
按 training.seed 设置随机种子，选择 device
按 data.name 创建 train_transform、eval_transform；过拟合时关闭增强
选择 build_mnist_loaders 或 build_cifar10_loaders，创建三个 loader
按 model.name 创建模型，然后 model.to(device)
创建 CrossEntropyLoss
按 training.optimizer 创建 Adam 或 AdamW
按 training.seed 重置训练随机状态
如果 smoke：调用 run_smoke，然后 return
如果 overfit_samples 不是 None：调用 run_overfit，然后 return
调用 run_training
```

两个检查模式互斥，由已有 parse_args 保证。每次命令都新建模型与优化器，正式训练从新进程开始。这里重置随机状态发生在模型初始化之后、读取第一批训练数据之前；DataLoader 的独立 Generator 仍由数据函数按 seed 创建。以后 Day 5 读取初始权重时，安排在创建优化器和这次重置之前。

检查 main 中三个模式调用的实参，是否与 3.2 的函数签名逐项对应，尤其不要漏掉 criterion。epoch 循环在 run_training 中；后面的初始状态、逐轮日志和资源记录也加到该函数的对应位置。先保持对象创建过程可直接阅读；哪一段以后明显变长，再单独提取那一段。

**3.7 同步独立评估入口，再检查两条入口。**

在 `scripts/evaluate.py` 顶层补充 CNN、build_cifar10_loaders 和 v2 的导入。删除“仅支持 mlp / mnist”的固定拒绝条件，保留 dtype 与 checkpoint 的核对，然后扩展已有对象创建过程：

1. 保留原来的 model_config、数据名称、validation_size、normalize、seed、split hash 核对；CIFAR 再核对 image_size。核对完成后才能重建并评估。
2. 取 `model_cfg = checkpoint["model_config"]`，按 name 创建 MLP 或 CNN。构造参数与训练入口一致，随后加载完整 state_dict，移到本次设备并切 eval。
3. 按 data.name 创建确定性 eval_transform：MNIST 使用 ToTensor → Normalize，CIFAR 使用 ToImage → ToDtype → Normalize，并检查 image_size=32。mean/std 读取已核对的当前配置。
4. 选择对应加载函数，使用 3.4 的同一套公共参数，但 train_transform 和 eval_transform 两个位置都传 eval_transform；接收三个 loader 后按 --split 选择 val 或 test。
5. 统计完整 split，核对 n_samples 等于对应 dataset 的长度。数据名称、模型名称不支持时，在相应分支报错。

这里保留少量相同的构造语句；训练和评估的一致性靠相同配置、checkpoint 核对和独立验证来检查。若以后预处理或模型创建规则变得复杂，再把需要共同维护的那部分单独提取。

Day 6 收集验证误分类时再扩展展示：MNIST 使用单通道灰度画法，CIFAR 使用三通道 mean/std 逆归一化，再转为 `[H,W,3]`。本周验收使用 val，测试指标留到第五周。

按下面顺序回归；前四条用于确认整理后 MNIST 行为一致，后两条用于接通新模型。MNIST 独立验证使用第二周已有 best，核对完整 5000 条与原记录，避免用重新训练覆盖旧结果：

```bash
python -m pytest tests/test_cv_training.py tests/test_datasets.py -q
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --smoke
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --overfit-samples 64
python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split val
python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --smoke
python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --overfit-samples 64
```

自查 smoke 只取少量批次，overfit 反复使用关闭增强后一次取出的固定张量；检查模式结束后，正式配置、history 和 checkpoint 不应变化。通过本步后继续第四步的 CNN / BN 检查，再补齐第五、六步的正式结果管理，最后启动完整训练。

#### 第四步：先检查再正式训练

创建 `tests/test_resnet.py`，今天先放 CNN 检查，Day 3–4 再向同一文件添加 BasicBlock 与 ResNet18 检查。顶层导入 torch、CNN；分别用 B=1/4 的 `[B,3,32,32]` 图片检查 `[B,10]` 输出，再用 long 标签计算交叉熵并 backward，核对活跃参数梯度存在且有限。

先打印各阶段尺寸，再 smoke、64 样本过拟合、正式训练。验证切 eval；若模型在 eval 前后差异异常，检查 BN running statistics 是否更新、加载是否完整。

BatchNorm2d 对每个通道沿 batch 与空间位置统计；例如 `[B,32,32,32]` 有 32 组统计量。训练时使用当前批次的统计，并更新 running mean/variance；默认评估时使用保存的运行统计。因此 `no_grad()` 不能替代 `eval()`，完整保存和加载 state_dict 时也要包含这些 buffer。

一组通道内先计算 `(x-mean)/sqrt(var+eps)`，再乘可学习的 gamma、加 beta；统计量与可学习参数是两类对象。默认 momentum=0.1 时，运行均值更新为 `0.9*旧均值+0.1*本批均值`，这里的 momentum 与优化器的动量不是同一个公式。若手算运行方差，注意训练前向用的方差与更新 running_var 用的方差估计不同，不能直接拿同一个值逐项对照，详见 [BatchNorm2d 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)。本周先理解统计范围和 train/eval 行为，不要求另写 BN 实现。

在现有检查文件中用一个独立 `BatchNorm2d(2)` 验证：输入 `[2,2,4,4]` 的非零均值数据，先 clone running_mean、running_var 和 num_batches_tracked；train 下前向一次，计数应加 1，运行统计按当前数据更新；再切 eval，用同一输入前向，两次 eval 输出一致且三个 buffer 都不再改变。用 `named_parameters()` 查 weight/bias，用 `named_buffers()` 查运行统计，不要求 train 和 eval 的输出在任何输入上都必然不同。

<a id="initial-state"></a>
#### 第五步：在训练前保存可供 Day 5 复用的初始状态

检查和测时结束后，按 seed=42 重新创建正式模型；模型尚未参加训练或更新 BN 统计时，立即保存 initial.pt。保存应早于第一次训练前向和 optimizer.step，不能将 best/last 改名作为初始权重。

在 `train_cv.py` 的 `run_training` 内，校验 epochs、计算 split hash 并保存配置后，为本周 CIFAR 正式运行加入以下保存步骤；位置在 epoch 循环之前。用 data.name=cifar10 的条件限定这段新增保存逻辑。`cfg` 为配置字典，`model` 是 main 传入的刚创建、尚未训练的模型：

```python
from pathlib import Path
import torch

initial_path = Path(cfg["output_dir"]) / "initial.pt"
initial_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "model_config": cfg["model"],
    "seed": cfg["training"]["seed"],
    "step": 0,
}, initial_path)
```

state_dict 包含参数与 BN buffers；此处直接写盘，不保留会随训练改变的内存引用。初始状态保存由 `run_training` 执行，smoke/overfit 由各自函数处理。后续 off 组加载并对齐权重后，也在各自 output_dir 保存自己的 initial.pt。

模型初始化或初始权重加载完成后，再按 training.seed 重置训练随机状态，并创建训练 loader 的迭代器。每对实验采用相同的数据采样与增强设置，避免模型构造消耗随机数的差异影响后续数据顺序。

<a id="training-resources"></a>
#### 第六步：接入逐轮日志与实际资源记录

在已有 `scripts/train_cv.py` 顶层补充 json、time；CSV 工具需要的 csv 和 Path 位于 `src/petvision/utils/history.py`。本步的循环和调用位置均修改 `run_training`。第二周 MNIST 的五轮小实验保持循环结束后统一保存。本周 CIFAR 训练较长，改为每个 epoch 完成后只追加当前 row，训练结束后再从完整 history 绘图；不要把累计 history 整个传给追加写入。

在 `src/petvision/utils/history.py` 顶层新增 `append_history(row, experiment_dir)`，与 save_history、plot_history 放在一起；然后将 train_cv.py 的导入扩为 `from petvision.utils.history import save_history, append_history, plot_history`。两个保存函数统一使用第二周的五列：epoch、train_loss、train_accuracy、val_loss、val_accuracy。

先把 save_history 中原来的局部 columns 提到模块顶层，命名为 HISTORY_COLUMNS，并将其中 DictWriter 的 fieldnames 改为 HISTORY_COLUMNS。append_history 也引用同一份列名，保证一次性保存和逐轮追加的字段、顺序完全一致。下面展示公共列名与新增函数：

```python
HISTORY_COLUMNS = [
    "epoch", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy",
]

def append_history(row, experiment_dir):
    path = Path(experiment_dir) / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

variant 是实验组标签，由 run_training 取 `Path(cfg["experiment_dir"]).name`，例如 cnn_bn_on。每组 history 已保存到独立的 runs/<variant> 目录，因此单组的逐轮记录保持上述五列；variant 用于曲线标题和 Day 6 的 results.csv 汇总表。向两个保存函数传入的 row 都保持五个字段。

从零实验应使用新的运行目录；若重跑同一 variant，先归档原配置、history、曲线和权重，再开始一份空 history。

在 run_training 写入配置和 initial.pt 之前，检查本组 history.csv 是否已存在且非空；CIFAR 正式运行遇到这种情况时明确报错，提示先归档或改用新目录，避免把两次从零训练的记录追加到同一个文件。

在 `run_training` 的 for 循环前设置 `record_resources = cfg["data"]["name"] == "cifar10"`，并取 variant 供绘图使用。两种数据共用已有训练 / 验证循环，按下面顺序增加保存操作：

1. 每轮按第二周结构构造包含 epoch 和四项指标的 row，MNIST 与 CIFAR 使用相同字段。
2. 执行 history.append(row)。CIFAR 接着调用 append_history(row, experiment_dir)，MNIST 此时只保留内存中的 history。
3. 每轮继续按原来的规则保存 last，并在验证准确率严格提升时保存 best。
4. 循环结束后，MNIST 调用一次 save_history；CIFAR 直接使用已经逐轮追加的 CSV。
5. 两者最后都调用 plot_history。在 history.py 中将它的签名扩为 `plot_history(history, experiment_dir, variant=None)`，CIFAR 调用时额外传 variant，函数内据此设置总标题；MNIST 继续使用原来的两个参数。

保留一个训练 / 验证循环，数据差异集中在日志与资源记录的位置。两轮临时数据检查 CSV 应恰好一行表头、两行数据，表头与 save_history 的五列完全一致。

训练总耗时在模型 / loader / 优化器准备完、initial.pt 保存后开始，覆盖所有训练、验证、逐轮日志和 best/last 保存；排除下载、初始化和最终画图。以下计时与 summary 代码仅在 record_resources 为真时执行，位置都在 `run_training` 内。先在正式 for 循环前执行：

```python
if device.type == "cuda":
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
started = time.perf_counter()
```

for 循环结束后、画图前执行：

```python
if device.type == "cuda":
    torch.cuda.synchronize(device)
train_seconds = time.perf_counter() - started
peak_allocated_mib = (
    torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "cuda" else None
)
summary = {
    "params": sum(p.numel() for p in model.parameters()),
    "epochs": cfg["training"]["epochs"],
    "train_seconds": train_seconds,
    "peak_allocated_mib": peak_allocated_mib,
    "device": str(device),
    "dtype": str(next(model.parameters()).dtype),
    "timing_scope": "train+validation+epoch_logging+checkpoint_io",
}
with (experiment_dir / "run_summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
```

CUDA allocated 峰值包含已加载模型等当前 Torch 分配，不等于整张显卡使用量；CPU 对应值为 null。计时使用 [time.perf_counter](https://docs.python.org/zh-cn/3.12/library/time.html#time.perf_counter)，GPU 前后同步使用 [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)。用于确定预算的单 epoch 短运行也按同样方法测量，但结果只打印，正式 summary 来自重新初始化后的完整运行。Day 6 的 train_seconds 从各自 run_summary.json 读取，不能靠回忆填写。

GPU 运算可以异步提交，Python 执行到下一行时 GPU 可能还没算完；synchronize 会等待已提交运算完成，让起止计时包含实际计算。reset_peak_memory_stats 重置的是“峰值记录”，不释放模型或数据。max_memory_allocated 返回字节数，除以 `1024**2` 换算成 MiB；因此字段写 peak_allocated_mib，而不是没有单位的 memory。

完成第五、六步后，按下面的顺序核对 `run_training`，避免把保存和计时插入错误的位置：

```text
检查 epochs，计算 split hash，准备实验目录与配置快照
建立空 history，设置 best_accuracy 和 record_resources
CIFAR：保存尚未训练的 initial.pt，开始计时与显存峰值记录
循环每个 epoch：训练 → 验证 → 建立 row → 记录 history → 保存 last / 更新 best
CIFAR：结束计时，保存 run_summary.json
MNIST：统一保存 history.csv
两者：根据本次内存中的 history 绘图
```

### 运行与验收

```bash
python -m pytest tests/test_resnet.py -q
python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --smoke
python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml --overfit-samples 64
python scripts/train_cv.py --config configs/cv/cnn_cifar10.yaml
```

今天的检查文件先只包含 CNN 检查，后两天逐步增加残差块和完整网络检查。前三条通过、预算确定后才执行最后一条；正式训练应先写出 step=0 的 initial，再保存该轮 history、曲线和 best/last。默认训练 20 epochs，若测时后统一调整预算，将实际值写入配置。

- [ ] CNN logits、loss、梯度正确，小样本能够拟合。
- [ ] 能解释 BN 对 batch/channel 的统计与评估行为。
- [ ] 训练器复用第二周流程，正式训练有真实配置与验证日志。

<a id="day-3"></a>
## Day 3：手写 BasicBlock 与投影分支

<a id="resnet-reading"></a>
### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [《动手学深度学习》7.6 残差网络](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)（主材料） | 7.6.2「残差块」的结构图、PyTorch `Residual` 类与两个形状示例 | 画出两条路径，再手算本项目的输入输出 |
| [李沐：29 残差网络 ResNet](https://www.bilibili.com/video/BV1bV41177ap/)（配套视频，按需观看） | 「ResNet」讲解与「代码」分段，重点看单个残差块 | 对照图文指出主分支、旁路和相加位置 |

今天先学单个块，按下面顺序阅读；这些操作计入下方的 1h 阅读时间。图文能读懂时直接跟做，需要口头讲解时穿插视频对应部分。

1. 打开教材的 7.6.2，代码标签选择 **PyTorch**。先看残差块示意图，找出输入、主分支、旁路和相加节点。在本项目中把主分支记作 `F(x)`，块的输出写成 `ReLU(F(x) + shortcut(x))`。
2. 先读 `Residual.forward`，逐行跟踪主分支和旁路，确认相加发生在最后一次 ReLU 之前。再回到 `__init__`，找到 forward 用到的两层卷积、两个 BN 和可选的 1×1 卷积；把每个层对应到图上的位置。
3. 对照本页第二步的例 A、例 B，在纸上分别写出两条路径的出口形状。先判断输入能否直接相加；不能时，再确定旁路的输出通道与 stride。
4. 按下表把教材名称对应到项目接口，再开始第一步的实现。自查时先口头说明每个参数控制什么，再把具体形状代入。

| 教材中的名称 | 本项目中的对应项 |
| --- | --- |
| `Residual` | `BasicBlock`，本日要实现的基本残差块 |
| `input_channels` / `num_channels` / `strides` | `in_channels` / `out_channels` / `stride` |
| `conv3` | 本项目命名为 `shortcut`，保存旁路上的 1×1 投影；不用投影时为 None |
| `use_1x1conv` | 保留同名参数，只控制是否创建投影；项目默认 True，需要恒等旁路时显式传 False，教材的默认值为 False |
| 教材始终执行残差相加 | Day 3–4 保持相同计算顺序；Day 5 再增加独立的 `use_residual` 开关 |

公式中的 shortcut(x) 表示整条旁路的结果；代码中的 self.shortcut 只保存可选投影。它为 None 时，旁路结果直接使用 x，forward 不调用 None。

今天按教材的计算顺序实现：投影分支只有 1×1 卷积，各卷积沿用默认的 `bias=True`，所有块都执行残差相加。项目将 `use_1x1conv` 默认设为 True，省略时会创建投影，即使输入 / 输出通道相同、stride=1 也一样。需要直接保留输入时显式传 False，此时输入与主分支输出必须同形状；通道或尺寸变化时使用 True，让旁路先完成投影。Day 4 沿用这个块，按教材组装整网并检查前反向与各阶段形状；Day 5 再增加用于消融的 `use_residual`。

7.6.1「函数类」可在理解基本块后选读；7.6.3「ResNet模型」留到 Day 4。残差块练习沿用本项目的 PyTorch 环境，教材的 `d2l` 训练封装和数据加载不引入项目。[原论文](https://arxiv.org/abs/1512.03385)作为实现后的补充阅读，先看残差块示意图和第 3 节感兴趣的部分；[torchvision 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html)作为拓展，用来理解两种实现的区别。

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 按上述顺序读图与教学实现，手算两条路径的尺寸 |
| 2h | 实现普通块和 1×1 projection |
| 1h | 检查形状、梯度和错误尺寸 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `src/petvision/models/cv/resnet.py` 的 BasicBlock：3×3 Conv→BN→ReLU→3×3 Conv→BN，加旁路后 ReLU。今天所有块都执行残差相加。

2. 默认 `use_1x1conv=True`，使用 1×1 Conv projection；同通道、stride=1 且需要恒等旁路时显式传 False，旁路直接使用原输入。先手算 `[2,64,32,32]→[2,128,16,16]` 的两条路径。

3. 分别测试普通块、只改变通道、只下采样、同时改变通道与尺寸，覆盖 batch=1/2。对参与前向的参数检查梯度存在且有限。

4. 解释 identity 与 projection 各适用什么条件，故意构造一次不匹配的相加并阅读报错，再修复。检查只打印，不要求额外 CSV。

### 分步操作与自查

#### 第一步：分别实现主分支和 shortcut

创建 `src/petvision/models/cv/resnet.py`，顶层写 `import torch` 和 `from torch import nn`，在其中定义继承 `nn.Module` 的 `BasicBlock(in_channels, out_channels, stride=1, use_1x1conv=True)`。接口沿用教材的投影开关，项目约定默认 True，步幅参数命名为 stride；调用时用关键字明确传入这两个参数。今天先实现单块，明天在同一文件增加 ResNet18；测试从 `petvision.models.cv.resnet` 导入 BasicBlock。

主分支按两层 3×3 卷积写出，第一层负责 stride；第二层保持尺寸。第二层 BN 的输出直接进入残差相加，最后再 ReLU，与教材的 forward 顺序一致。

具体按“先注册层，再连接数据”的顺序完成：

1. 在构造函数创建两层 Conv 及对应的 BN；用 `nn.ReLU()` 创建激活模块的实例，保存为 self.relu，在两个激活位置复用。各层只创建一次，不放进 forward。
2. `use_1x1conv=True` 时创建旁路 1×1 卷积，保存到 `self.shortcut`；False 时令其为 None。先在纸上分别写出两条路径的出口形状，确认能够逐元素相加。
3. 在 forward 中用 `identity = x` 保留原始输入，让另一变量 out 依次经过 Conv1 → BN1 → ReLU → Conv2 → BN2，得到主分支输出。
4. `self.shortcut is not None` 时，将原始 x 送入投影并更新 identity；否则 identity 保持为 x。两种情况都将 out 与 identity 相加，不能把主分支的中间结果送入投影。
5. 对相加结果执行最后一次 ReLU 并返回。再用下一步的两个输入逐段打印形状，定位哪一层第一次偏离手算。

两层主分支卷积都用 padding=1；第一层将 in_channels 变为 out_channels，使用传入 stride，第二层接收并输出 out_channels，stride=1。投影使用 `Conv2d(in_channels, out_channels, 1, stride=stride)`，今天各卷积保持教材默认的 bias=True。这个 1×1 卷积称为 projection（投影）：它调整通道数，并按相同步幅缩小高宽，使两条路径能逐元素相加。旁路为 None 时表示恒等路径，仍然参与相加。

```text
x ── Conv/BN/ReLU ── Conv/BN ── + ── ReLU
└──────── identity/projection ──┘
```

#### 第二步：处理两个具体输入

例 A：输入 `[2,64,32,32]`，创建 `BasicBlock(64, 64, use_1x1conv=False, stride=1)`；主分支与原输入都为 `[2,64,32,32]`，直接相加。

例 B：同样输入，创建 `BasicBlock(64, 128, use_1x1conv=True, stride=2)`；主分支为 `[2,128,16,16]`，shortcut 的 1×1 Conv 也用 stride=2，输出必须一致。

再用 `BasicBlock(64, 128, stride=2)` 检查默认调用，省略开关后仍使用投影，输出形状与例 B 一致。例 A 的 False 需要显式保留，才能检查直接使用原输入的恒等旁路。

下面是本文件 forward 的旁路与相加部分；先完成主分支计算，令 out 保存第二层 BN 的输出，再接上这段：

```python
identity = x
if self.shortcut is not None:
    identity = self.shortcut(x)
out = out + identity
return self.relu(out)
```

`if` 只决定是否投影，`out + identity` 始终执行。故意把例 B 的 `use_1x1conv` 改成 False，观察直接相加的尺寸错误，再改回 True；关闭投影不会关闭残差。

#### 第三步：写出能暴露错误的检查

把下面各项添加到 Day 2 已创建的 `tests/test_resnet.py`，使用 `test_` 开头的函数名，保留已有 CNN 检查。

对 batch=1/2 分别覆盖下面四种情况；输入均为 `[B,64,32,32]`，它们是单个块的随机特征图，整网的 96×96 图片输入在 Day 4 检查。对输出求标量 loss 再反向，检查主分支和存在的投影分支参数都有有限梯度。

| out_channels | stride | use_1x1conv | 预期输出 |
| --- | --- | --- | --- |
| 64 | 1 | False | `[B,64,32,32]` |
| 128 | 1 | True | `[B,128,32,32]` |
| 64 | 2 | True | `[B,64,16,16]` |
| 128 | 2 | True | `[B,128,16,16]` |

若相加报尺寸错误，分别打印 out.shape 和 identity.shape；有投影时继续检查 shortcut(x).shape。先确认调用时的 use_1x1conv，再核对两条路径的 stride。

增加一个能确认残差确实参与相加的确定性小例子：同尺寸、显式传入 use_1x1conv=False 的块，x 全为 1，主分支两层卷积的权重、卷积 bias 及 BN bias 均设为 0，并切 eval。主分支输出应为 0，块的输出应全为 1。这个检查使用恒等旁路，保持显式的 False。随后另建普通随机块做有限梯度检查；置零例子专门验证相加路径，不用于要求每个梯度都非零。Day 5 再在这个例子上补残差关闭后的输出检查。

#### 第四步：自己解释一次

说明不用投影时为什么仍然要相加，为什么通道或空间尺寸变化时需要投影，以及主分支和旁路分别怎样得到最终形状。此时先掌握标准块；关闭残差的比较放到 Day 5。

### 运行与验收

```bash
python -m pytest tests/test_resnet.py -q
```

补充的 BasicBlock 检查应覆盖上表四种情况；例 B 的输出应为 `[2,128,16,16]`。若残差相加报错，先分别打印两条路径的形状，检查投影是否启用，以及主分支第一层与 projection 的 stride 是否一致。

- [ ] 两条路径相加前形状一致，下采样块输出正确。
- [ ] use_1x1conv 只控制投影，标准块始终执行残差相加。
- [ ] 能解释为什么 stride 或通道变化需要 projection。

<a id="day-4"></a>
## Day 4：按教材组装 ResNet18 并检查整网

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [《动手学深度学习》7.6 残差网络](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)（主材料） | PyTorch 版的 7.6.2 Residual、7.6.3 resnet_block 与 b1–b5，以及 7.6.4 的 resize=96 | 将教材的灰度入口改为 RGB，再逐段核对形状 |
| [Resize 官方文档](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html) | 二元组指定目标高宽、antialias 参数 | 在训练与评估入口生成 96×96 图片 |
| [ResNet 原论文](https://arxiv.org/abs/1512.03385)（实现后选读） | 表 1 的 18 层结构与残差块示意图 | 对照自己的网络数出 18 层 |

本日以教材的教学实现作为结构参考，检查自己模型的整网前反向与各阶段形状。教材训练 Fashion-MNIST 时将单通道图片从 28×28 放大到 96×96，前面打印各层形状的随机输入则是 224×224。项目使用同一套残差结构和 96×96 训练尺寸，输入通道改为 CIFAR-10 的 3；数据划分、训练器、AdamW 和结果保存继续使用项目入口。

教材的卷积使用默认 bias=True，投影旁路只有 1×1 Conv。torchvision 的卷积 bias 与投影 BN 设置不同，其源码作为拓展选读；本日的测试直接调用自己实现的 ResNet18。

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 对照教材理解入口、stage 和 96×96 输入 |
| 2h | 组装网络，接通配置与 Resize |
| 1h | 检查整网前反向与各阶段形状 |
| 0.5h | smoke、小批检查与启动正式基线 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 在 `src/petvision/models/cv/resnet.py` 沿用 Day 3 的块：卷积 bias=True，投影只有 1×1 Conv；按教材添加 stage 辅助函数和 ResNet18，使用 7×7 入口卷积及最大池化。
2. 创建 `configs/cv/resnet_cifar10.yaml`，设 model.name=resnet18、num_classes=10，data.image_size 按 CIFAR-10 原图填写 32；在 train_cv.py 和 evaluate.py 接入 ResNet18，并为它添加一致的 Resize(96,96)。
3. 扩展已有 `tests/test_resnet.py`，检查 batch=1/2 的整网前反向，以及 b1–b5、全局池化、展平和分类层的输出形状。
4. 通过短检查、冻结残差两组预算后，重新创建正式模型，在首次训练前保存 `outputs/resnet_on/initial.pt`，再运行残差 on 基线。

### 分步操作与自查

#### 第一步：按教材组织入口与四个 stage

继续修改 `src/petvision/models/cv/resnet.py`。先核对 BasicBlock：两层主分支卷积和存在的投影卷积都使用 bias=True，shortcut 为单个 1×1 Conv 或 None。如果当前写成了 bias=False 或在投影后接 BN，先调整为上述教材结构，再运行 Day 3 的块检查。

接着在同一文件添加 `resnet_block(in_channels, out_channels, num_residuals, first_block=False)`，用于创建一个 stage，按下面顺序处理：

1. 建立一个空列表，按 num_residuals 循环，每次新建一个 BasicBlock。
2. 如果是该 stage 的第一个块且 first_block=False，创建 in_channels→out_channels、stride=2、use_1x1conv=True 的块。
3. 其他位置创建 out_channels→out_channels、stride=1、use_1x1conv=False 的块。first_block=True 用于 stage1，其输入和输出通道均为 64。
4. 返回块列表；在下面组装网络时用 `nn.Sequential(*块列表)` 注册这些块。列表用于准备内容，实际调用和参数注册由 Sequential 完成。

项目的 use_1x1conv 默认 True，所以普通块必须明确传 False；教材在普通块处省略参数，依赖的是它自己的默认 False。照着教材组织循环时保留这个接口差异。

在同一文件定义 `ResNet18(num_classes=10)`，构造函数中依次建立 b1–b5，再保存到 self.net：

1. b1 是入口 Sequential：3→64 的 7×7 Conv，stride=2、padding=3、bias=True，接 BatchNorm2d(64)、ReLU、3×3 MaxPool2d(stride=2,padding=1)。
2. b2 是 stage1：调用辅助函数，通道 64→64、块数 2、first_block=True。
3. b3、b4、b5 分别是 64→128、128→256、256→512，块数均为 2、first_block=False。各 stage 仅首块下采样。
4. 用一个 Sequential 将 b1、b2、b3、b4、b5、AdaptiveAvgPool2d((1,1))、Flatten(start_dim=1)、Linear(512,num_classes) 按顺序保存到 self.net。
5. forward 调用 self.net 并返回 logits。先用随机 `[B,3,96,96]` 输入核对下表；use_residual 在 Day 5 再加入。

| self.net 索引 | 教材分组 / 操作 | 预期输出 |
| --- | --- | --- |
| 输入 | CIFAR-10 缩放后的 RGB 图片 | `[B,3,96,96]` |
| 0 | b1：入口卷积、BN、ReLU、最大池化 | `[B,64,24,24]` |
| 1 | b2：stage1，两个普通块 | `[B,64,24,24]` |
| 2 | b3：stage2，首块下采样 | `[B,128,12,12]` |
| 3 | b4：stage3，首块下采样 | `[B,256,6,6]` |
| 4 | b5：stage4，首块下采样 | `[B,512,3,3]` |
| 5 | 全局平均池化 | `[B,512,1,1]` |
| 6 | 展平 | `[B,512]` |
| 7 | 分类头 | `[B,10]` |

b1 内部先由入口卷积将高宽从 96 变为 48，再由最大池化变为 24。stage2 的首块例如 `BasicBlock(64, 128, stride=2, use_1x1conv=True)`，把 `[B,64,24,24]` 变成 `[B,128,12,12]`；后一个普通块保持该形状。ResNet18 的层数按入口 1 层卷积、8 个块各 2 层卷积、最后 1 层 Linear 计算。

<a id="d2l-resnet-input"></a>
#### 第二步：让训练和评估入口生成 96×96 输入

继续修改 `scripts/train_cv.py` 和 `scripts/evaluate.py` 的 CIFAR 分支。data.image_size 按进入预处理流程的数据集图像尺寸填写：CIFAR-10 原图是 32×32，因此 CNN 和 ResNet 的配置都写 32，沿用已有 CIFAR 分支的检查。ResNet 额外 Resize 到 96×96；这个目标尺寸由本节的模型预处理分支决定，不回写为原图尺寸。

1. 训练入口先按原规则在 32×32 图片上执行随机裁剪与翻转。RandomCrop 的裁剪尺寸取 data.image_size，本例仍是 32，padding 来自配置；不要将裁剪尺寸改成 Resize 的目标 96，否则 32×32 图片加 padding=4 后仍不够裁剪。
2. 完成增强后，若模型名称为 resnet18，添加 `v2.Resize((96, 96), antialias=True)`；随后执行原来的 ToDtype(torch.float32,scale=True) 和 Normalize。CNN 分支不添加这一步 Resize。模型名称与第三步的配置、构造分支保持一致。
3. eval_transform 使用 ToImage、按模型分支需要添加的 Resize、ToDtype、Normalize。训练入口的验证 loader 与独立 evaluate.py 使用相同的确定性顺序。
4. 可以先用列表准备这些操作，按条件 append Resize，再交给 v2.Compose。两套 transform 继续传给已有 build_cifar10_loaders；图像大小的转换由 transform 负责。
5. 过拟合模式继续让 train_transform 使用 eval_transform，此时输入仍为 96×96，只是关闭随机增强。
6. evaluate.py 保留 data.image_size 与 checkpoint 配置的相等校验，两者都应记录原图尺寸 32；再按保存的模型名称选择确定性预处理。CNN 实际输入为 32×32，ResNet 两组实际输入为 96×96，不能把原图尺寸直接当作所有模型的最终输入尺寸。

ResNet 训练顺序是 ToImage → RandomCrop(32,padding=4) → RandomHorizontalFlip → Resize(96,96) → ToDtype → Normalize；验证 / 测试顺序是 ToImage → Resize(96,96) → ToDtype → Normalize。RGB mean/std 继续使用三项 0.5；这是缩放原来的图片，类别标签和固定划分不变。

#### 第三步：接入模型选择和配置

继续修改 `scripts/train_cv.py` 和 `scripts/evaluate.py`，按下面顺序接入模型：

1. 在两个文件顶层从 `petvision.models.cv.resnet` 导入 ResNet18。
2. 在现有 model.name 选择处增加 `resnet18` 分支，调用 `ResNet18(num_classes=model_cfg["num_classes"])`。训练入口使用当前配置中的 model 字段，评估入口继续使用 checkpoint 保存的 model_config；两处构造参数一致。
3. ResNet18 的 7×7 入口卷积、最大池化和残差块结构已经固定在类中，配置不增加 stem 字段，入口也不读取或校验该字段。今天只传 num_classes；Day 5 再增加可选的 residual 开关。
4. 核对上一步的 Resize 条件同样判断 `model.name == "resnet18"`。配置名称、模型构造和预处理使用同一个字符串，避免构造了 ResNet18 却遗漏 Resize。

从 CNN 配置完整复制出 `configs/cv/resnet_cifar10.yaml`，保留数据路径、划分、归一化、增强以及训练字段，再修改下面列出的字段：

```yaml
model:
  name: resnet18
  num_classes: 10

data:
  image_size: 32  # CIFAR-10 原图为 32x32；ResNet 的 Resize 在预处理分支完成

experiment_dir: experiments/cnn_vs_resnet/week3/runs/resnet_on
output_dir: outputs/resnet_on
```

这是字段修改清单，不是独立完整配置。model 部分只保留本模型需要的字段，复制时移除 CNN 专用的 use_bn；其余 data 和 training 字段继续保留。image_size 按数据集输入图像的实际高宽填写，本例仍为 32；它不用于选择网络入口结构，也不代表经过 Resize 后的最终张量尺寸。

接线后分别读取一批训练和验证数据，逐项检查：

1. 原始 CIFAR 图片为 32×32，配置中的 data.image_size=32。
2. ResNet 分支的预处理包含 Resize(96,96)，images 为 float32、形状 `[B,3,96,96]`；标签为 long 且在 0–9 内。
3. 模型名称为 resnet18，输出 logits 为 `[B,10]`；训练和独立评估都使用相同的预处理与构造规则。
4. 再用已有 CNN 配置做 smoke，确认其输入仍为 `[B,3,32,32]`。

暂时只做这些短检查，后面通过模型测试后才启动 ResNet 正式训练。

#### 第四步：检查整网前反向与各阶段形状

继续扩展已有 `tests/test_resnet.py`，保留 Day 2–3 的 CNN、BN 和 BasicBlock 检查；在顶层导入自己的 ResNet18。本日使用随机的 `[B,3,96,96]` 输入，分别添加以下两个检查函数。

先实现 `test_resnet_forward_backward`，检查整网能否完成一次分类计算和反向传播：

1. 遍历 batch=1、2，每次重新创建 CPU 上的 ResNet18 并切换为 train，避免共用上一次的梯度和 BN 运行统计。
2. 创建对应批次的随机图片，以及值在 0–9 内、dtype 为 long 的类别标签。
3. 清除梯度后调用模型，断言 logits 形状为 `[B,10]`，且所有值有限。
4. 用交叉熵计算 loss，断言它是有限标量，再执行 backward。
5. 遍历 named_parameters，断言各参数的 grad 非 None 且全部有限；报错时带上参数名称，便于定位入口、残差块、投影旁路或分类头。

再实现 `test_resnet_stage_shapes`，检查下采样发生在正确位置：

1. 重新创建 CPU 上的 ResNet18，切换为 eval，并准备 batch=2 的随机图片。
2. 按第一步的形状表建立“model.net 索引 → 预期 shape”的字典，覆盖 b1–b5、全局平均池化、展平和分类层共 8 个模块；先检查模块数量一致。
3. 在 no_grad 下按顺序遍历 model.net，每次将上一层输出送入下一层，打印模块索引、类型与实际 shape。
4. 对每层立即断言实际 shape 与预期一致，错误信息同时列出模块索引、预期值和实际值，定位最先发生偏差的位置。

前反向检查和阶段形状检查通过后，再进行下一步的短训练。

#### 第五步：保存初始状态并启动基线

先执行下方的 smoke 和固定 64 样本过拟合检查，按 96×96 输入测量一轮实际耗时，再冻结残差 on/off 的 epoch、batch 等预算。CNN 与 ResNet 的分辨率不同，记录实际 image_size 和耗时，跨架构比较按完整配置解释。

短检查结束后重新创建正式模型和优化器，按 Day 2 的规则先保存 step=0 的 `outputs/resnet_on/initial.pt`，再开始正式训练；短训练参数和 BN 统计只属于检查模型。Day 5 将复用这份教材结构的初始状态。

### 运行与验收

完成本日模型、配置和两个入口的修改后执行：

```bash
python -m pytest tests/test_resnet.py -q
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml --smoke
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml --overfit-samples 64
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml
```

前三条通过且预算冻结后，再执行最后一条。整网检查应打印各阶段尺寸，并在断言失败时指出具体的模块索引或参数名称。

- [ ] CIFAR RGB 图片经过正确的增强 / Resize 顺序，训练与评估输入均为 96×96。
- [ ] 7×7 入口、最大池化、bias=True 和单卷积投影与教材结构对应，普通块明确关闭投影。
- [ ] batch=1/2 的整网前反向与各阶段形状检查通过。
- [ ] 正式训练前保存 initial.pt，CNN 32×32 分支仍可复用。

<a id="day-5"></a>
## Day 5：BN 与残差的单变量对照

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [BatchNorm2d 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.BatchNorm2d.html) | 可训练仿射参数和运行统计 | 明确 BN 对照差异 |
| [ResNet 原论文](https://arxiv.org/abs/1512.03385)（选读） | plain / residual 的比较思路；实验操作按本页步骤完成 | 只从同深宽对照解释残差 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 写下两组实验假设与控制条件 |
| 2h | 接入并检查残差开关，补充配置、对齐初始化并运行 |
| 1h | 核对训练日志和参数量 |
| 0.5h | 整理配置快照与图表 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 创建 `cnn_no_bn_cifar10.yaml` 和 `resnet_no_residual_cifar10.yaml`，分别指定唯一输出目录。四份配置的 experiment_dir 使用 `experiments/cnn_vs_resnet/week3/runs/<variant>` 保存各自 history 与配置快照，output_dir 使用独立 `outputs/<variant>` 保存 checkpoint，避免日志相互覆盖。四份配置明确数据划分、增强、预算与种子。

2. BN 对照使用同一 CNN，只将 BN 替换为 Identity；两组卷积 bias 和其他层保持一致，从 Day 2 保存的 initial.pt 复制并核对共有参数的初始值。

3. 在已通过整网前反向与阶段形状检查的网络上增加独立的 use_residual，沿用同一主分支深度与宽度。关闭时跳过旁路计算和相加，保留投影模块的定义与 state_dict 键；从 Day 4 保存的 initial.pt 复制初始状态，核对默认开启时的行为保持一致。

4. 完成两组 off 训练，与 on 组使用同样划分、优化器、epoch、batch 和增强。保留每组配置快照、history 和 best，不能把浅层 CNN 与 ResNet 的差异直接当残差结论。

### 分步操作与自查

<a id="residual-ablation"></a>
#### 第一步：接入残差开关，再把配置按两对整理

修改已完成的 `src/petvision/models/cv/resnet.py`，在标准块上增加实验开关。use_1x1conv 继续只控制是否需要投影，use_residual 单独控制是否计算旁路并相加；两者使用不同名称。

1. 将块接口扩展为 `BasicBlock(in_channels, out_channels, stride=1, use_1x1conv=True, use_residual=True)`，在构造函数保存 self.use_residual。use_1x1conv 继续默认 True；保持主分支、ReLU 位置和各层名称不变。
2. 将整网接口扩展为 `ResNet18(num_classes=10, use_residual=True)`，stage 辅助函数 resnet_block 也增加同名参数。整网调用四次 resnet_block 时传入开关，辅助函数再把它传给全部八个 BasicBlock；各块原来的 use_1x1conv 和 stride 保持不变。
3. 投影模块仍只根据 use_1x1conv 创建，与 use_residual 无关。这样 on/off 的模型参数和 buffer 键集合相同，Day 4 的状态可以继续加载；关闭时保留投影的参数，但跳过它的计算。
4. forward 先照常计算主分支 out；仅在 use_residual=True 时准备 identity、按需投影并执行相加。False 时直接对主分支输出执行最后一次 ReLU。下方片段接在第二层 BN 之后，替换 Day 3 的旁路与相加部分。

```python
if self.use_residual:
    identity = x
    if self.shortcut is not None:
        identity = self.shortcut(x)
    out = out + identity
return self.relu(out)
```

5. 修改 `scripts/train_cv.py` 和 `scripts/evaluate.py` 的 resnet18 构造分支：从各自使用的 model_config 中读取 `.get("residual", True)`，检查结果为布尔值，再传给 ResNet18 的 use_residual。缺省视为开启，使 Day 4 未填写该字段的配置与 checkpoint 仍可使用；训练和评估采用相同规则。
6. 在已有 `tests/test_resnet.py` 补开关检查。保留 Day 4 的整网前反向和阶段形状检查，确认 use_residual=True 时仍然通过，新增开关不产生参数或 buffer。对普通块和投影块分别检查 on/off 的输出形状与主分支有限梯度；off 时投影的 weight、bias 不参与计算，它们的 grad 应为 None。梯度检查使用新模型或先将旧梯度清为 None。另在本地交互环境创建默认开启的 ResNet18，用已有 checkpoint 工具读取 Day 4 的 initial.pt，将 model_state 完整加载并检查无缺失或多余 key；这项自查使用本地实验文件，pytest 检查继续用临时状态，避免依赖仓库外的权重。
7. 扩展 Day 3 的置零例子：同尺寸、use_1x1conv=False、x 全为 1，主分支输出为 0 并切 eval。开启残差时输出全为 1，关闭时输出全为 0；再另建普通随机块检查有限梯度。运行 `python -m pytest tests/test_resnet.py -q`，通过后再准备 off 组训练。

此方案保留投影模块，因此残差两组的总参数量相同；off 组的投影参数不参与前向和反向。run_summary.json 的 params 继续表示总参数量，比较时另说明这一计算路径差异。

开关检查通过后，把已有 on 配置与待创建的 off 配置按两对整理：

```text
BN 对照：    cnn_bn_on  ↔ cnn_bn_off
残差对照：   resnet_on  ↔ resnet_off
```

每对共享划分、增强、训练预算、优化器、batch 和共有层初始权重。两个实验问题分别分析，不用跨对差异解释某一个开关。

#### 第二步：核对共有参数的初始值

在训练入口支持可选的 `training.init_checkpoint` 字段：on 组留空或省略；off 组按下表读取已经保存的 step=0 状态。该字段用于从初始权重开始新的实验，进度仍从 0 起、优化器重新创建。

| off 配置 | training.init_checkpoint |
| --- | --- |
| `cnn_no_bn_cifar10.yaml` | `outputs/cnn_bn_on/initial.pt` |
| `resnet_no_residual_cifar10.yaml` | `outputs/resnet_on/initial.pt` |

复制对应 on 配置后，BN off 只将 `model.use_bn` 改为 false，残差 off 在 model 中添加 `residual: false`；分别添加上表的 `training.init_checkpoint`。Day 4 的 on 配置保持原样，省略 residual 即按 True 重建，保留它与已保存 model_config 的一致性。两份新配置的 experiment_dir 改为本周 `runs/cnn_bn_off`、`runs/resnet_off`，output_dir 改为 `outputs/cnn_bn_off`、`outputs/resnet_off`。不要覆盖原 on 组文件。

先确认文件存在且 step=0，再创建 off 模型，逐 key 复制共有参数和适用的 buffer；共有层的 shape 与值都要检查。BN off 用 Identity 替换 BN 时应保持其他层命名稳定，只允许预期的 BN key 差异。残差 off 保留投影模块，因此与 on 组的模型状态键应完全相同，包含投影卷积的 weight、bias；任何漏项或形状不符都应先修复。

复制完成后，保存 off 组自己的 initial.pt，并与来源文件逐项核对共有参数；列出 BN 对照中独有的 BN key，残差对照的独有模型状态 key 应为空。不以 strict=False 没有报错代替检查。来源文件缺失时先补齐初始状态及对应基线，不能临时用训练后的 best/last 代替。

实际核对时先列出两个 state_dict 的共有 key、来源独有 key、目标独有 key。共有 key 必须 shape 相同，复制后用 `torch.equal` 检查完全一致；BN 对照的独有 key 逐项归入预期 BN 差异，残差对照检查全部模型状态一致。最终打印共有/独有 key 数与最大共有参数差，后者应为 0；训练前的 BN running buffers 也按同一规则处理。

仅设置相同 seed 不一定保证共有层初值一致：新增/删除随机初始化层会改变后续随机数消费顺序。对齐后新建优化器，按 Day 2 约定重置训练随机状态，再取第一批数据。

数据顺序应由独立 DataLoader Generator 控制；相同 seed 本身不保证不同网络消费完全相同的随机数。若要逐样本对齐随机增强，还需独立控制增强随机流或保存增强决策；未做时只声称增强策略和种子一致，并在单次实验局限中说明。

#### 第三步：确认输出不会互相覆盖

四个 variant 各自有 experiment_dir、history、best、last；曲线标题标明 variant，单组 history 使用统一五列，Day 6 的 results.csv 汇总行再写入 variant。训练保存配置的方法沿用 Day 2 的 run_training，逐轮日志见[资源记录步骤](#training-resources)。再将四份实际使用的 YAML 各复制一份到 `experiments/cnn_vs_resnet/week3/configs/`，沿用原配置文件名，供本周 README 集中索引；复制内容应与各 runs/<variant>/config.yaml 一致。若中途改变 batch 或 epoch，要把该对两组都按新条件重跑。

#### 第四步：运行并检查训练现象

按两对实验分别完成以下检查：

1. 先确认 on 组已有对应 initial.pt、正式 history、best/last 和配置快照，off 组将使用独立输出目录。缺少初始状态时先处理第二步，不用训练后的权重顶替。
2. 对 off 组运行形状检查和短检查，核对载入的共有参数、实际增强和样本数。短检查结束后丢弃临时模型，再按初始状态开始正式运行。
3. 正式运行第一轮结束后，打开该组 history.csv，确认只有一个表头、epoch=1 的一行，loss / accuracy 有限；同时确认其他组日志未被修改。
4. 运行到约定预算后，检查 epoch 连续且行数正确，读取最后一行并核对 last；再按验证选择规则核对 best。某次中断未完成预算时先标记为未完成，不能与完整运行直接作结论。
5. 将同一对的起始、中间和最后阶段放在一起看：训练 loss 是否下降、验证是否同步改善、train/val 差距是否增大。若有 NaN 或异常跳变，先定位发生的 epoch，再检查输入、梯度与优化设置。

比较 BN 或残差时，结果不必一定支持预设假设；不支持也保存曲线，并记录单种子、小预算及残差 off 跳过投影计算带来的局限。

### 运行与验收

确认两份来源 initial 已存在且共有参数核对通过，再运行两组新实验：

```bash
python scripts/train_cv.py --config configs/cv/cnn_no_bn_cifar10.yaml
python scripts/train_cv.py --config configs/cv/resnet_no_residual_cifar10.yaml
```

Day 2 和 Day 4 的 on 组已经完成时直接复用，不为填写对照表再训练一次。off 组虽然读取了初始权重，仍从零进度、新优化器开始；这与第七周恢复训练的语义不同。

- [ ] BN 和残差各有清楚假设、控制条件与参数量记录。
- [ ] use_1x1conv 与 use_residual 的职责独立，默认开启残差时仍通过 Day 4 整网检查，旧配置与状态可加载。
- [ ] on/off 的 initial.pt 均为 step=0，每对共有主分支初始化一致，输出目录与 checkpoint 不互相覆盖。
- [ ] 四组均完成冻结预算，短运行与正式结果有明确区分。

<a id="day-6"></a>
## Day 6：验证结果、错误分析与周总结

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [CIFAR-10 教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/blitz/cifar10_tutorial.html) | 类别预测与错误检查 | 分析验证误分类 |
| [总计划实验规范](StudyPlan.md#acceptance) | 对照、验证与最终测试口径 | 整理真实结果 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 汇总四组验证指标和曲线 |
| 1.5h | 分析训练现象与误分类 |
| 1h | 有限参考重写一个残差块 |
| 0.5h | 更新本周 README |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 当天任务概览

1. 将每组最佳验证 loss/accuracy、best epoch、参数量、训练耗时、配置路径与 checkpoint 写入 `experiments/cnn_vs_resnet/week3/results.csv`；注明 split=val。

2. 分别画 BN on/off 与 residual on/off 的 train/val 曲线，查看少量验证错误案例；先描述事实，再说明是否支持假设。单种子、小预算结果限定结论范围。

3. 有限参考下重写一个支持下采样的 BasicBlock，解释两条路径的每个尺寸和残差的梯度通路。

4. 更新周 README，链接固定划分、四组配置、曲线与结果；将四个 checkpoint 及版本标识留给第五周，不使用 CIFAR 测试集挑模型。

### 分步操作与自查

#### 第一步：整理一张可检查的验证表

将四组汇总结果保存为 `experiments/cnn_vs_resnet/week3/results.csv`。在 Python 草稿或交互环境读取各组 history 和独立验证结果，整理下面的字段，再按第一周 csv.DictWriter 方法创建父目录、写入表头和四行真实结果；不需要另建必做脚本。

每行至少包括 variant、split=val、image_size、best_epoch、val_loss、val_accuracy、params、epochs、batch、train_seconds、config、checkpoint。结果表的 image_size 记录实际送入模型的尺寸：先从预处理后的 images.shape 核对并保存到该组运行摘要，再由汇总读取；CNN 两组为 32，ResNet 两组为 96。它与 config 中记录原图尺寸的 data.image_size 分别说明，不能直接复制后者填入结果表。

params 和 train_seconds 读取各组 experiment_dir/run_summary.json，epoch 指标读取同目录 history.csv；batch、epochs 来自对应 config.yaml。独立验证先打印并核对完整结果；本周汇总表由这里明确列出的文件与该次验证输出填写。

best 的指标来自同一个验证选择规则。最终一轮指标和 best epoch 的指标可以不同，汇总时写清选择口径。

按各组保存的 best epoch 从自身 history 找同一行，使用这一行的 val_loss 和 val_accuracy；二者不能分别取各列的最优值。用下方独立评估命令覆盖全部 5000 条验证样本，再核对这两个指标。CSV 写法复用第一周 Day 6，曲线读取各组自己的 history。

#### 第二步：两张对照图分别回答一个问题

BN 图只画同 CNN 两组的 train/val；残差图只画同深宽两组。横轴 epoch，纵轴 loss 或 accuracy，标图例、单位与训练预算。

读取四个 runs/<variant>/history.csv，用第一周的 subplots / plot / savefig 方法分别保存 `experiments/cnn_vs_resnet/week3/figures/bn_comparison.png` 和 `experiments/cnn_vs_resnet/week3/figures/residual_comparison.png`。图与 results.csv 来自已有训练记录，不为绘图重新训练；在本周 README 链接这两张图和结果表。

选少量验证误分类，注明真实类和预测类。对照结论依次写观察事实、可能解释、控制条件和局限。

将第二周 evaluate.py 的 collect_mistakes 也接到 `--split val` 分支；收集后按三通道 mean/std 逆归一化、permute(1,2,0) 显示。图片分别保存到各组 `experiment_dir/figures/mistakes_val.png`；测试图用 mistakes_test.png，避免后续测试覆盖验证分析。这次只取验证图片，完整验证指标仍由全体 5000 条统计。

#### 第三步：有限参考重写 BasicBlock

先写形状，再写标准残差块的 forward；分别验证 use_1x1conv=False 的普通块与 use_1x1conv=True、stride=2 的下采样块，均保留残差相加。能解释为什么第二个 3×3 Conv 后要先完成残差相加再激活。

将草稿中的重写块实际实例化，输入例 A、例 B，打印输出形状并对标量目标 backward，检查活跃参数梯度有限；再与 Day 3 的已验证块对照 forward 顺序。已有 `test_resnet.py` 的通过只覆盖被它调用的实现，重写草稿也要做这两次小张量运行。

#### 第四步：交接第五周所需材料

1. 打开本周 README，按四个 variant 各列一行，链接其 config.yaml、history.csv、run_summary.json 和 best 路径；区分本周汇总表与各运行目录。
2. 逐项打开链接或检查路径，确认没有把 on 组权重填到 off 组，也没有把归档记录误当本轮结果。
3. 核对四组均引用同一 cifar10_split.json，保存文件及其 hash；第五周直接复用这份索引，不重新抽取验证集。
4. 在新进程中用一组配置重建并加载 best，完整运行验证，确认返回 5000 条及匹配的指标；其余组应已有本日第一步的独立验证记录。
5. 将仍缺少的材料写成明确待办，例如“resnet_off 缺 run_summary”；补齐后再在第五周使用该组。测试集继续保留到第五周统一评估。

### 运行与验收

使用第二周建立的独立评估入口核对四个 best，全部只评估验证集：

```bash
python scripts/evaluate.py --config configs/cv/cnn_cifar10.yaml --checkpoint outputs/cnn_bn_on/best.pt --split val
python scripts/evaluate.py --config configs/cv/cnn_no_bn_cifar10.yaml --checkpoint outputs/cnn_bn_off/best.pt --split val
python scripts/evaluate.py --config configs/cv/resnet_cifar10.yaml --checkpoint outputs/resnet_on/best.pt --split val
python scripts/evaluate.py --config configs/cv/resnet_no_residual_cifar10.yaml --checkpoint outputs/resnet_off/best.pt --split val
```

每组应覆盖 `5000` 条验证样本，结果与自身 best 所在 epoch 一致。将四行真实结果写入本周 `results.csv`，保存两类对照图；本周不运行 `--split test`。

- [ ] 两类对照均有结果、曲线、解释与局限。
- [ ] 所有指标清楚标记 val；测试集仍保留。
- [ ] 固定划分、预处理、配置与 best 可在第五周直接复用。

## 本周结束检查

- [ ] CNN、BasicBlock、ResNet18 的形状、梯度与教材结构入口检查通过。
- [ ] BN 与同深宽残差开关两类对照完成，控制条件和参数量清楚。
- [ ] 固定 CIFAR 划分、四组 checkpoint、验证曲线与报告可复用。
- [ ] 本周没有用测试集调参，70%+ 仅留作第五周 CNN 测试的参考目标。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第四周](week4.md)。

