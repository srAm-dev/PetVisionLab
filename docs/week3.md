# 第三周逐日学习指南：CNN、ResNet 与 BN / 残差对照

> 更新日期：2026-10-27<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-1)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：手写 CNN、BasicBlock 和 CIFAR 版 ResNet18，完成 BN on/off 与同深宽残差 on/off 两类对照，保留统一划分供第五周复用。  
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
│   └── models/cv/resnet.py                 # Day 3–4：BasicBlock、ResNet18
├── configs/cv/
│   ├── cnn_cifar10.yaml                    # Day 2：BN on
│   ├── cnn_no_bn_cifar10.yaml              # Day 5：BN off
│   ├── resnet_cifar10.yaml                 # Day 4：残差 on
│   └── resnet_no_residual_cifar10.yaml     # Day 5：残差 off
├── scripts/inspect_cifar10.py              # Day 1：数据、划分与形状
├── scripts/train_cv.py                     # Day 2–5：扩展模型选择，沿用入口
├── tests/test_resnet.py                    # Day 2–4：形状、反向、参考对照
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

CIFAR-10 官方 50,000 条训练数据固定划分 45,000/5,000，种子 42，索引保存到 `experiments/cnn_vs_resnet/week3/cifar10_split.json`，第五周原样复用。从本周开始，自己构建的分类图像预处理统一使用 `torchvision.transforms.v2`；第八周预训练 CLIP/VLM 使用各自配套的 processor。训练顺序为 ToImage → RandomCrop(32,padding=4) → RandomHorizontalFlip → ToDtype(torch.float32,scale=True) → Normalize，固定 mean/std=(0.5,0.5,0.5)；验证和测试只用 ToImage → ToDtype → Normalize。分别创建训练/验证 dataset 实例再套固定索引，避免共享随机增强。

CNN 起步为三层 3×3 Conv，通道 32/64/128，前两层后各做 2×2 Pool，最后 GAP→Linear(128,10)；BN 开关只替换归一化，同一对中卷积 bias 均保持 True。ResNet18 用 [2,2,2,2] 个 BasicBlock、通道 64/128/256/512，CIFAR stem=3×3,stride=1,padding=1，没有初始 maxpool。四组从零训练，起始 batch=64、AdamW lr=1e-3、weight_decay=1e-4、20 epochs、FP32；先短运行测时，再冻结两组对照的预算。显存不足可统一减小同一对的 batch，正式结果需注明预算变化。测试集保留到第五周。

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
| Day 3 | 手写 BasicBlock 与投影分支 | 4h | 1h | 同尺寸与下采样残差块 |
| Day 4 | 组装 CIFAR 版 ResNet18 并对照参考 | 4h | 1h | ResNet18、参考层对齐检查 |
| Day 5 | BN 与残差的单变量对照 | 4h | 1h | 四组训练、配置快照与验证记录 |
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

### 具体怎么做

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
#### 第二步：下载 CIFAR-10，明确加载函数与预处理

今天把第二周的 MNIST 数据加载流程扩展到 **CIFAR-10**：三通道 32×32 图片、10 类，官方训练集 50000 张、测试集 10000 张。在 `src/petvision/data/datasets.py` 新增 `build_cifar10_loaders(data_dir, split_path, batch_size, seed, train_transform, val_transform, num_workers=0, validation_size=5000)`，返回 train_loader、val_loader、test_loader。两个 transform 由调用者传入，便于正式训练使用增强、过拟合检查改用确定性变换；validation_size 和 num_workers 都由配置传入，不能只接收 batch_size。

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
val_transform = v2.Compose([
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
val_base = CIFAR10(root=data_dir, train=True, download=True, transform=val_transform)
test_set = CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
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
| 2h | 实现 CNN、模型选择和 smoke |
| 1h | 小批过拟合并启动正式基线 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `src/petvision/models/cv/cnn.py`，按本周结构实现可开关 BN 的 CNN；每个阶段打印一次形状，最终输出 `[B,10]`。

2. 扩展 `train_cv.py` 的模型与数据选择，创建 `configs/cv/cnn_cifar10.yaml`。沿用第二周的按样本加权指标、best checkpoint、history 和种子处理。

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

修改已有 `scripts/train_cv.py`，顶层从 `petvision.models.cv.cnn` 导入 CNN，从 `petvision.data.datasets` 导入 build_cifar10_loaders。具体接线沿用[第二周对象创建顺序](week2.md#training-objects)：data.name=mnist 时保留原函数，data.name=cifar10 时先按配置创建两个 transform，再调用 Day 1 的 build_cifar10_loaders；model.name=cnn 时创建 CNN(num_classes=..., use_bn=...)。training.optimizer=adamw 时创建 torch.optim.AdamW 并传入配置的学习率、weight_decay。已有 `scripts/evaluate.py` 使用同一套模型 / 数据选择规则；过拟合检查把传给训练 dataset 的 transform 改为确定性 val_transform。

同时把第二周“仅支持 mlp / mnist / adam”的入口校验改成实际已实现的选择分支；未知名称仍报错。只新增 CNN 创建代码而保留旧的拒绝条件，会在创建模型之前退出。当前 dtype 仍限定 float32；CIFAR 的 image_size 当前只支持 32，先明确校验，不能接收其他值却继续按 32 训练。

把 Day 1 的固定 transform 改为读取 data.normalize 的 mean/std、augmentation.random_crop_padding 和 random_horizontal_flip；在 train_cv.py 和 evaluate.py 顶层均导入 `from torchvision.transforms import v2`。两个变换仍用 v2.Compose：训练先 ToImage，再按配置加入 v2.RandomCrop / v2.RandomHorizontalFlip，最后 ToDtype(torch.float32,scale=True) 和 v2.Normalize。padding=0 时省略随机裁剪，flip=false 时省略翻转；flip=true 时使用 RandomHorizontalFlip 默认概率 0.5，不表示每张都翻转。val_transform 始终只包含 v2.ToImage、v2.ToDtype(torch.float32,scale=True) 和 v2.Normalize。下面是 main 中 CIFAR 分支的调用，两个 transform 已先建立：

```python
data_cfg = cfg["data"]
train_loader, val_loader, test_loader = build_cifar10_loaders(
    data_dir=data_cfg["root"],
    split_path=data_cfg["split_path"],
    batch_size=data_cfg["batch_size"],
    seed=cfg["training"]["seed"],
    train_transform=train_transform,
    val_transform=val_transform,
    num_workers=data_cfg["num_workers"],
    validation_size=data_cfg["validation_size"],
)
```

过拟合模式在调用前令 train_transform=val_transform；独立评估也使用确定性变换。检查三个 loader 的大小、实际 transform 和一个批次，而不是只打印 YAML。第五周 ViT 继续复用这条 CIFAR 分支。

#### 第三步：先检查再正式训练

创建 `tests/test_resnet.py`，今天先放 CNN 检查，Day 3–4 再向同一文件添加 BasicBlock 与 ResNet18 检查。顶层导入 torch、CNN；分别用 B=1/4 的 `[B,3,32,32]` 图片检查 `[B,10]` 输出，再用 long 标签计算交叉熵并 backward，核对活跃参数梯度存在且有限。

先打印各阶段尺寸，再 smoke、64 样本过拟合、正式训练。验证切 eval；若模型在 eval 前后差异异常，检查 BN running statistics 是否更新、加载是否完整。

BatchNorm2d 对每个通道沿 batch 与空间位置统计；例如 `[B,32,32,32]` 有 32 组统计量。训练时使用当前批次的统计，并更新 running mean/variance；默认评估时使用保存的运行统计。因此 `no_grad()` 不能替代 `eval()`，完整保存和加载 state_dict 时也要包含这些 buffer。

在现有检查文件中用一个独立 `BatchNorm2d(2)` 验证：输入 `[2,2,4,4]` 的非零均值数据，先 clone running_mean、running_var 和 num_batches_tracked；train 下前向一次，计数应加 1，运行统计按当前数据更新；再切 eval，用同一输入前向，两次 eval 输出一致且三个 buffer 都不再改变。用 `named_parameters()` 查 weight/bias，用 `named_buffers()` 查运行统计，不要求 train 和 eval 的输出在任何输入上都必然不同。

<a id="initial-state"></a>
#### 第四步：在训练前保存可供 Day 5 复用的初始状态

检查和测时结束后，按 seed=42 重新创建正式模型；模型尚未参加训练或更新 BN 统计时，立即保存 initial.pt。保存应早于第一次训练前向和 optimizer.step，不能将 best/last 改名作为初始权重。

在正式训练入口中加入以下保存步骤，`cfg` 为配置字典，`model` 为刚创建、尚未训练的模型：

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

state_dict 包含参数与 BN buffers；此处直接写盘，不保留会随训练改变的内存引用。保存逻辑只进入正式训练分支，smoke/overfit 仍只打印。后续 off 组加载并对齐权重后，也在各自 output_dir 保存自己的 initial.pt。

模型初始化或初始权重加载完成后，再按 training.seed 重置训练随机状态，并创建训练 loader 的迭代器。每对实验采用相同的数据采样与增强设置，避免模型构造消耗随机数的差异影响后续数据顺序。

<a id="training-resources"></a>
#### 第五步：接入逐轮日志与实际资源记录

在已有 `scripts/train_cv.py` 顶层补充 csv、json、time；结果仍由正式分支生成。第二周 MNIST 的五轮小实验保持循环结束后统一保存。本周 CIFAR 训练较长，改为每个 epoch 完成后只追加当前 row，训练结束后再从完整 history 绘图；不要把累计 history 整个传给追加写入。

在脚本顶层新增 `append_history(row, experiment_dir)`。本周 row 包含 variant 和第二周五项 epoch 指标，variant 取 `Path(cfg["experiment_dir"]).name`。下面是该函数的实现：

```python
def append_history(row, experiment_dir):
    path = Path(experiment_dir) / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    columns = ["variant", "epoch", "train_loss", "train_accuracy",
               "val_loss", "val_accuracy"]
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
```

从零实验应使用新的运行目录；若重跑同一 variant，先归档原配置、history、曲线和权重，再开始一份空 history。每轮先 history.append(row)，再 append_history(row, experiment_dir)；结束后 plot_history(history, experiment_dir)。CIFAR 分支不再调用旧 save_history 覆盖追加结果；MNIST 保留原逻辑。两轮临时数据检查 CSV 应恰好一行表头、两行数据，variant 全部一致。

训练总耗时在模型 / loader / 优化器准备完、initial.pt 保存后开始，覆盖所有训练、验证、逐轮日志和 best/last 保存；排除下载、初始化和最终画图。在正式 for 循环前执行：

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

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [ResNet 原论文](https://arxiv.org/abs/1512.03385) | 第 3 节、残差块示意图与 identity shortcut | 画出 F(x)+shortcut(x) |
| [torchvision ResNet 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html) | BasicBlock、downsample、forward | 手写形状对齐 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读残差公式与 BasicBlock |
| 2h | 实现普通块和 1×1 projection |
| 1h | 检查形状、梯度和错误尺寸 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `src/petvision/models/cv/resnet.py` 的 BasicBlock：3×3 Conv→BN→ReLU→3×3 Conv→BN，加 shortcut 后 ReLU。实现 `use_residual` 开关，关闭时主分支计算保持不变。

2. 当 stride=2 或通道数不同，残差 on 使用 1×1 Conv+BN projection；同尺寸使用 identity。先手算 `[2,64,32,32]→[2,128,16,16]` 的两条路径。

3. 分别测试同通道 stride=1、变通道 stride=2、batch=1、residual off。对参与前向的参数检查有限梯度，投影只在启用残差时参与计算。

4. 解释 identity 与 projection 各适用什么条件，故意构造一次不匹配的相加并阅读报错，再修复。检查只打印，不要求额外 CSV。

### 分步操作与自查

#### 第一步：分别实现主分支和 shortcut

创建 `src/petvision/models/cv/resnet.py`，顶层写 `import torch` 和 `from torch import nn`，在其中定义 `BasicBlock(in_channels, out_channels, stride=1, use_residual=True)`。今天先实现单块，明天在同一文件增加 ResNet18；测试从 `petvision.models.cv.resnet` 导入 BasicBlock。

主分支按两层 3×3 卷积写出，第一层负责 stride；第二层保持尺寸。BN 后的 ReLU 位置要与参考一致，第二层输出在残差相加前不先 ReLU。

两层卷积都用 padding=1；第一层将 in_channels 变为 out_channels，使用传入 stride，第二层接收并输出 out_channels，stride=1。旁路 shortcut 在通道相同且 stride=1 时用 Identity；否则用 `Conv2d(in_channels,out_channels,1,stride=stride,bias=False)` 加 `BatchNorm2d(out_channels)`。这个 1×1 卷积称为 projection（投影）：它调整通道数，并按相同步幅缩小高宽，使两条路径能逐元素相加。主分支卷积也用 bias=False，与 Day4 的 ResNet 参考保持一致。

```text
x ── Conv/BN/ReLU ── Conv/BN ── + ── ReLU
└──────── identity/projection ──┘
```

#### 第二步：处理两个具体输入

例 A：`[2,64,32,32]`、out_channels=64、stride=1，主分支与 shortcut 都为 `[2,64,32,32]`。

例 B：同样输入、out_channels=128、stride=2，主分支为 `[2,128,16,16]`；shortcut 的 1×1 Conv 也用 stride=2，输出必须一致。

```python
def forward(self, x):
    # 先算主分支 out
    ...
    if self.use_residual:
        out = out + self.shortcut(x)
    return self.relu(out)
```

#### 第三步：写出能暴露错误的检查

把下面各项添加到 Day 2 已创建的 `tests/test_resnet.py`，使用 `test_` 开头的函数名，保留已有 CNN 检查。

分别覆盖 stride=1/2、通道相同/变化、batch=1/2、residual on/off。对输出求标量 loss 再反向，检查参与路径的梯度。

若相加报尺寸错误，分别打印 out.shape 与 shortcut(x).shape；若关闭残差仍执行相加，开关测试应失败。检查时不要求未参与路径产生梯度。

增加一个能区分开关的确定性小例子：同尺寸 identity 块、x 全为 1，主分支两层卷积的权重、存在的卷积 bias 及 BN bias 均设为 0，并切 eval。主分支输出应为 0，residual on 输出应全为 1，off 应全为 0。随后另建普通随机块做有限梯度检查；上述置零例子专门验证相加路径，不用于要求每个梯度都非零。

#### 第四步：自己解释一次

说明 residual off 的主分支仍有相同层数与宽度，projection 的参数是否保留由实现决定并需记录；浅层 CNN 不是该开关实验的另一组。

### 运行与验收

```bash
python -m pytest tests/test_resnet.py -q
```

补充的 BasicBlock 检查应覆盖普通块、下采样块与残差关闭；例 B 的输出应为 `[2,128,16,16]`。若残差相加报错，先分别打印两条路径的形状，检查主分支第一层与 projection 的 stride 是否一致。

- [ ] 两条路径相加前形状一致，下采样块输出正确。
- [ ] 残差开关只改变 shortcut 路径与相加，主分支深度宽度一致。
- [ ] 能解释为什么 stride 或通道变化需要 projection。

<a id="day-4"></a>
## Day 4：组装 CIFAR 版 ResNet18 并对照参考

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [torchvision ResNet 源码](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html) | ResNet、_make_layer、resnet18 默认 stem | 同步修改参考输入层 |
| [ResNet 原论文](https://arxiv.org/abs/1512.03385) | 18 层结构与不同 stage 下采样 | 核对 [2,2,2,2] |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 复习 stage 数量与下采样位置 |
| 2h | 组装网络并修改 torchvision 参考 |
| 1h | 权重、形状与数值对齐 |
| 0.5h | smoke 和正式残差 on 训练 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 完成 ResNet18：CIFAR stem 输出 `[B,64,32,32]`，四个 stage 空间尺寸依次 32、16、8、4，再 AdaptiveAvgPool→flatten→Linear(512,10)。

2. 建立 `torchvision.models.resnet18(weights=None,num_classes=10)` 参考；将 conv1 换为 3×3,stride=1,padding=1，maxpool 换 Identity。结构对照时使用 residual on。

3. 逐项对齐 state_dict，包括 BN 的可训练参数与 running buffers；若键名不同则显式映射并检查覆盖率，不能忽略未匹配主分支。相同输入、eval、FP32 下比较 logits，起始容差 1e-5，并分别验证反向有限。

4. 创建 `configs/cv/resnet_cifar10.yaml`，记录参数量和 epoch 耗时，通过 smoke/小批检查后重建正式模型，按 Day 2 的保存规则写 `outputs/resnet_on/initial.pt`，再运行残差 on 基线。

### 分步操作与自查

#### 第一步：按 stage 数量组织 ResNet18

继续修改 `src/petvision/models/cv/resnet.py`，在 BasicBlock 之外定义 `ResNet18(num_classes=10, use_residual=True)`，按下表组合各 stage。扩展 `tests/test_resnet.py`，从该模块导入 ResNet18，形状与参考对照均在测试函数中执行，参考模型不会替代正式实现。

```text
stem               [B,64,32,32]
stage1: 2 blocks   [B,64,32,32]
stage2: 2 blocks   [B,128,16,16]
stage3: 2 blocks   [B,256,8,8]
stage4: 2 blocks   [B,512,4,4]
GAP + classifier   [B,10]
```

每个 stage 仅第一个 block 在需要时 stride=2；其余块 stride=1。GAP 后使用 flatten(start_dim=1)，保留 batch 维。

例如 stage2 的第一个块是 `BasicBlock(64,128,stride=2)`，把 `[B,64,32,32]` 变为 `[B,128,16,16]`；第二个是 `BasicBlock(128,128,stride=1)`，保持后者形状。stage1 不下采样，stage3/4 以同样方式在首块缩小高宽。ResNet18 的“18”通常按入口 1 层卷积、8 个块各 2 层卷积、最后 1 层 Linear 计数，不表示有 18 个 BasicBlock。

#### 第二步：同步修改参考网络

```python
reference = torchvision.models.resnet18(weights=None, num_classes=10)
reference.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
reference.maxpool = torch.nn.Identity()
```

自写 ResNet 的卷积 bias、BN eps、ReLU 位置和 shortcut 结构要与参考一致。此处的 bias=False 是 ResNet 的约定，前一天 CNN 的 BN 消融仍按固定 bias=True。

#### 第三步：权重对齐后再比较数值

先列两边 state_dict 的 key 和 shape，再复制所有可比参数和 buffer。映射时检查完整覆盖，不能只因 strict=False 没报错就认定对齐。

可沿用参考的 `conv1、bn1、layer1..layer4、fc` 和各块命名，让键名直接对应；若采用自定义名称，先建立「自写 key → 参考 key」字典。检查自写 state_dict 的每个 key 均被映射、参考端没有漏项，且每对 shape 相同，再在 no_grad 下复制。两个模型在 eval 下使用同一 CPU FP32 输入，输出最大绝对差以 `1e-5` 为起始容差；前反向检查使用各自新建输入与清空的梯度，不能把两边梯度累加到一起。

同一输入、eval、同 dtype/device 下计算 logits，打印最大绝对差与容差。差别大先查第一处不同的 stage 输出；随机初值各自不同的两个模型没有数值相等的前提。

#### 第四步：完成训练前核对

记录自写与参考参数量、stage 尺寸及前反向结果。结构核对通过后再做短训练；官方参考不替代正式手写模型。

正式模型必须重新初始化，不继承短训练的参数或 BN 统计。确认 `outputs/resnet_on/initial.pt` 的 step=0，再开始正式训练；Day 5 将从该文件复制共有主分支的初始状态。

从 CNN 配置复制出 `configs/cv/resnet_cifar10.yaml`，保留相同 data 和 training，模型与输出部分改为：

```yaml
model:
  name: resnet18
  num_classes: 10
  stem: cifar
  residual: true
experiment_dir: experiments/cnn_vs_resnet/week3/runs/resnet_on
output_dir: outputs/resnet_on
```

这段是要替换的字段，不是独立完整配置。入口将 `model.residual` 传给各块的 `use_residual`；`stem: cifar` 对应上面的 3×3 输入卷积和移除 maxpool。形状对照、参考权重复制和短运行均在检查模型上进行，正式模型单独重新创建。

在 `scripts/train_cv.py` 和 `scripts/evaluate.py` 顶层导入 ResNet18，新增 model.name=resnet18 分支，将 num_classes、residual 分别传给构造函数的 num_classes、use_residual；当前只支持 stem=cifar，其他值明确报错。训练仍复用原来的数据函数、训练器和 AdamW。

### 运行与验收

```bash
python -m pytest tests/test_resnet.py -q
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml --smoke
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml --overfit-samples 64
python scripts/train_cv.py --config configs/cv/resnet_cifar10.yaml
```

参考检查应打印或在断言失败时给出各 stage 尺寸、映射覆盖情况和 logits 最大误差。短检查通过后才运行正式残差 on 基线，并确认 `outputs/resnet_on/initial.pt` 与 best/last 的 step 含义不同。

- [ ] CIFAR stem 与参考同步修改，没有默认 7×7/池化混入比较。
- [ ] stage 输出尺寸和分类 logits 正确，权重与 buffer 对齐有证据。
- [ ] 能指出结构一致性检查与训练性能比较各证明什么。

<a id="day-5"></a>
## Day 5：BN 与残差的单变量对照

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [BatchNorm2d 文档（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.BatchNorm2d.html) | 可训练仿射参数和运行统计 | 明确 BN 对照差异 |
| [ResNet 原论文](https://arxiv.org/abs/1512.03385) | plain / residual 的比较思路 | 只从同深宽对照解释残差 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 写下两组实验假设与控制条件 |
| 2h | 补充两份配置、对齐初始化并运行 |
| 1h | 核对训练日志和参数量 |
| 0.5h | 整理配置快照与图表 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `cnn_no_bn_cifar10.yaml` 和 `resnet_no_residual_cifar10.yaml`，分别指定唯一输出目录。四份配置的 experiment_dir 使用 `experiments/cnn_vs_resnet/week3/runs/<variant>` 保存各自 history 与配置快照，output_dir 使用独立 `outputs/<variant>` 保存 checkpoint，避免日志相互覆盖。四份配置明确数据划分、增强、预算与种子。

2. BN 对照使用同一 CNN，只将 BN 替换为 Identity；两组卷积 bias 和其他层保持一致，从 Day 2 保存的 initial.pt 复制并核对共有参数的初始值。

3. 残差对照沿用同一主分支深度与宽度，关闭 shortcut 相加；若关闭组省略 projection，记录总参数量与参与训练参数的差异。从 Day 4 保存的 initial.pt 复制共有主分支初始权重，避免随机数消耗顺序导致初值不同。

4. 完成两组 off 训练，与 on 组使用同样划分、优化器、epoch、batch 和增强。保留每组配置快照、history 和 best，不能把浅层 CNN 与 ResNet 的差异直接当残差结论。

### 分步操作与自查

#### 第一步：把四份配置按两对整理

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

复制对应 on 配置后，BN off 只将 `model.use_bn` 改为 false，残差 off 只将 `model.residual` 改为 false；分别添加上表的 `training.init_checkpoint`。两份配置的 experiment_dir 改为本周 `runs/cnn_bn_off`、`runs/resnet_off`，output_dir 改为 `outputs/cnn_bn_off`、`outputs/resnet_off`。不要覆盖原 on 组文件。

先确认文件存在且 step=0，再创建 off 模型，逐 key 复制共有参数和适用的 buffer；共有层的 shape 与值都要检查。Identity 替换应保持其他层的命名稳定，键名不同时显式映射；只允许预期的 BN/projection 差异，其他漏配或形状不符应报错。

复制完成后，保存 off 组自己的 initial.pt，并与来源文件逐项核对共有参数；列出两边独有的 BN/projection key，不以 strict=False 没有报错代替检查。来源文件缺失时先补齐初始状态及对应基线，不能临时用训练后的 best/last 代替。

实际核对时先列出两个 state_dict 的共有 key、来源独有 key、目标独有 key。共有 key 必须 shape 相同，复制后用 `torch.equal` 检查完全一致；两边独有 key 逐项归入预期 BN 或 projection 差异，未能解释的条目应先修复。最终打印共有/独有 key 数与最大共有参数差，后者应为 0；训练前的 BN running buffers 也按同一规则处理。

仅设置相同 seed 不一定保证共有层初值一致：新增/删除随机初始化层会改变后续随机数消费顺序。对齐后新建优化器，按 Day 2 约定重置训练随机状态，再取第一批数据。

数据顺序应由独立 DataLoader Generator 控制；相同 seed 本身不保证不同网络消费完全相同的随机数。若要逐样本对齐随机增强，还需独立控制增强随机流或保存增强决策；未做时只声称增强策略和种子一致，并在单次实验局限中说明。

#### 第三步：确认输出不会互相覆盖

四个 variant 各自有 experiment_dir、history、best、last；曲线标题和 CSV 行写明 variant。训练保存配置的方法复用[第二周正式分支](week2.md#run-modes)。再将四份实际使用的 YAML 各复制一份到 `experiments/cnn_vs_resnet/week3/configs/`，沿用原配置文件名，供本周 README 集中索引；复制内容应与各 runs/<variant>/config.yaml 一致。若中途改变 batch 或 epoch，要把该对两组都按新条件重跑。

#### 第四步：运行并检查训练现象

正式结果至少查看起始、中间与最后几个 epoch。观察 train/val gap、梯度与收敛速度，先排查 loss 爆炸、数据与标签错误，再解释模型差异。

比较 BN 或残差时，结果不必一定支持预设假设；不支持也保存曲线，并记录单种子、小预算和 projection 差异带来的局限。

### 运行与验收

确认两份来源 initial 已存在且共有参数核对通过，再运行两组新实验：

```bash
python scripts/train_cv.py --config configs/cv/cnn_no_bn_cifar10.yaml
python scripts/train_cv.py --config configs/cv/resnet_no_residual_cifar10.yaml
```

Day 2 和 Day 4 的 on 组已经完成时直接复用，不为填写对照表再训练一次。off 组虽然读取了初始权重，仍从零进度、新优化器开始；这与第七周恢复训练的语义不同。

- [ ] BN 和残差各有清楚假设、控制条件与参数量记录。
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

### 具体怎么做

1. 将每组最佳验证 loss/accuracy、best epoch、参数量、训练耗时、配置路径与 checkpoint 写入 `experiments/cnn_vs_resnet/week3/results.csv`；注明 split=val。

2. 分别画 BN on/off 与 residual on/off 的 train/val 曲线，查看少量验证错误案例；先描述事实，再说明是否支持假设。单种子、小预算结果限定结论范围。

3. 有限参考下重写一个支持下采样的 BasicBlock，解释两条路径的每个尺寸和残差的梯度通路。

4. 更新周 README，链接固定划分、四组配置、曲线与结果；将四个 checkpoint 及版本标识留给第五周，不使用 CIFAR 测试集挑模型。

### 分步操作与自查

#### 第一步：整理一张可检查的验证表

将四组汇总结果保存为 `experiments/cnn_vs_resnet/week3/results.csv`。在 Python 草稿或交互环境读取各组 history 和独立验证结果，整理下面的字段，再按第一周 csv.DictWriter 方法创建父目录、写入表头和四行真实结果；不需要另建必做脚本。

每行至少包括 variant、split=val、best_epoch、val_loss、val_accuracy、params、epochs、batch、train_seconds、config、checkpoint。

params 和 train_seconds 读取各组 experiment_dir/run_summary.json，epoch 指标读取同目录 history.csv；batch、epochs 来自对应 config.yaml。独立验证先打印并核对完整结果；本周汇总表由这里明确列出的文件与该次验证输出填写。

best 的指标来自同一个验证选择规则。最终一轮指标和 best epoch 的指标可以不同，汇总时写清选择口径。

按各组保存的 best epoch 从自身 history 找同一行，使用这一行的 val_loss 和 val_accuracy；二者不能分别取各列的最优值。用下方独立评估命令覆盖全部 5000 条验证样本，再核对这两个指标。CSV 写法复用第一周 Day 6，曲线读取各组自己的 history。

#### 第二步：两张对照图分别回答一个问题

BN 图只画同 CNN 两组的 train/val；残差图只画同深宽两组。横轴 epoch，纵轴 loss 或 accuracy，标图例、单位与训练预算。

读取四个 runs/<variant>/history.csv，用第一周的 subplots / plot / savefig 方法分别保存 `experiments/cnn_vs_resnet/week3/figures/bn_comparison.png` 和 `experiments/cnn_vs_resnet/week3/figures/residual_comparison.png`。图与 results.csv 来自已有训练记录，不为绘图重新训练；在本周 README 链接这两张图和结果表。

选少量验证误分类，注明真实类和预测类。对照结论依次写观察事实、可能解释、控制条件和局限。

将第二周 evaluate.py 的 collect_mistakes 也接到 `--split val` 分支；收集后按三通道 mean/std 逆归一化、permute(1,2,0) 显示。图片分别保存到各组 `experiment_dir/figures/mistakes_val.png`；测试图用 mistakes_test.png，避免后续测试覆盖验证分析。这次只取验证图片，完整验证指标仍由全体 5000 条统计。

#### 第三步：有限参考重写 BasicBlock

先写形状，再写 forward；分别验证普通块和下采样块。能解释为什么第二个 3×3 Conv 后要先完成残差相加再激活。

将草稿中的重写块实际实例化，输入例 A、例 B，打印输出形状并对标量目标 backward，检查活跃参数梯度有限；再与 Day 3 的已验证块对照 forward 顺序。已有 `test_resnet.py` 的通过只覆盖被它调用的实现，重写草稿也要做这两次小张量运行。

#### 第四步：交接第五周所需材料

确认 cifar10_split.json、四组配置与 best 都能找到；读取一份 checkpoint 验证模型可重建。测试集仍未参与模型选择，测试指标按第五周统一安排。

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

- [ ] CNN、BasicBlock、ResNet18 的形状、梯度与 CIFAR stem 检查通过。
- [ ] BN 与同深宽残差开关两类对照完成，控制条件和参数量清楚。
- [ ] 固定 CIFAR 划分、四组 checkpoint、验证曲线与报告可复用。
- [ ] 本周没有用测试集调参，70%+ 仅留作第五周 CNN 测试的参考目标。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第四周](week4.md)。

