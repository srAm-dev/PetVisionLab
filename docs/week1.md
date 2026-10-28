# 第一周逐日学习指南：Python、NumPy、数学与线性回归

> 更新日期：2026-10-27<br>
> 对应计划：[StudyPlan.md 第一周](StudyPlan.md#stage-1)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 表示你的第一个学习日，可以从任意星期开始。  
> 本周成果：独立写出 NumPy 线性回归，解释预测、损失、梯度和参数更新，保存真实曲线，并在周末整理简短小结。  
> 环境：本地 CPU + Conda 命名环境 `petvision`，以 Python 3.12、NumPy、Matplotlib 起步；PyTorch 与 GPU 环境安排在第二周。

## 怎么使用这份指南

每天按“看指定内容 → 写练习 → 运行检查 → 解释结果”的顺序完成。学习资料只读当天列出的部分，其余内容后续再学。

本次已提供这份指南、[第一周说明与总结](../experiments/basics/week1/README.md)和 [当前本地环境快照](../environment.yml)。本指南按学习日提供环境准备、练习步骤、运行命令与自查；每周 README 保存实际环境、复现命令索引、结果和周总结。代码文件按任务创建，环境快照和实验结果在实际运行时保存；周末集中补充简短总结，代码由你亲手完成。每个脚本从项目根目录运行；代码中的函数名用于统一练习，具体实现留给你。

每小时适当休息。每天 1 小时机动用于语法、数学、报错和复盘；如果当天核心任务仍未通过验收，下一学习日先补齐。

## 本周目录蓝图

下图表示本周结束时逐步形成的目录，注释标明创建时间。生成的曲线、指标和环境快照来自实际运行。

```text
PetVisionLab/
├── README.md                          # Day 1：项目介绍与文档导航
├── .gitignore                         # Day 1：忽略本地环境和缓存
├── environment.yml                    # 已提供：当前本地 Linux 完整环境快照
├── docs/
│   ├── StudyPlan.md                   # 总学习计划
│   └── week1.md                       # 已提供：本指南
├── scripts/                           # Day 1 创建，脚本逐日增加
│   ├── python_basics.py               # Day 1：函数、循环、MSE
│   ├── numpy_basics.py                # Day 1 创建，Day 2 补充
│   ├── linear_model.py                # Day 2：类与模块练习
│   ├── math_basics.py                 # Day 3：矩阵与一维梯度下降
│   ├── gradient_basics.py             # Day 4：线性回归梯度
│   ├── linear_regression_numpy.py     # Day 5：完整训练
│   ├── gradient_check.py              # Day 6：有限差分检查
│   ├── learning_rate_comparison.py    # Day 6：学习率对照
│   └── linear_regression_rewrite.py   # Day 6：有限参考重写
└── experiments/
    └── basics/
        └── week1/
            ├── README.md             # 已提供：环境、运行说明与周总结
            ├── environment.txt       # Day 1：系统、Conda、Python、解释器路径
            ├── conda-packages.txt    # Day 1：实际安装包与版本
            ├── conda-explicit.txt    # Day 1：同平台 Conda 包精确快照
            ├── results.csv           # Day 6：真实实验结果
            └── figures/              # Day 5 创建
                ├── loss_curve.png    # Day 5：基准训练曲线
                └── learning_rates.png # Day 6：学习率对照
```

在编辑器文件树中按路径新建目录、文件即可。`scripts/` 用于代码，`experiments/basics/week1/` 保存环境快照、实验结果与周总结。

命名环境由 Conda 管理，项目保存定义和实际版本快照。每天开始学习时，先执行 `conda activate petvision`，再运行当天脚本；编辑器的 Python 解释器也选择该环境。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | 环境、Python、NumPy 起步 | 4h | 1h | 两个练习脚本与环境记录 |
| Day 2 | 数组、广播、类与模块 | 4h | 1h | 形状练习、LinearModel 类 |
| Day 3 | 线性代数、导数、梯度下降 | 4h | 1h | 形状自查与一维更新脚本 |
| Day 4 | 偏导、链式法则、MSE 梯度 | 4h | 1h | 梯度推导与单步更新 |
| Day 5 | NumPy 线性回归训练 | 4h | 1h | 完整脚本、loss 曲线 |
| Day 6 | 梯度检查、对照、重写与总结 | 4h | 1h | 学习率表、梯度检查、周总结 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：把环境和第一组练习跑起来

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Conda：环境管理](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) | 创建、激活环境与导出包清单，区分最小依赖和完整快照 | 建立 petvision 环境 |
| [Conda：包管理（中文译文）](https://docs.conda.org.cn/projects/conda/en/stable/user-guide/tasks/manage-pkgs.html) | 查看包版本、安装依赖 | 理解 Conda 管理 Python 与依赖的方式 |
| [Python：控制流](https://docs.python.org/zh-cn/3/tutorial/controlflow.html) | if、for、range、定义函数、return | 写循环和函数 |
| [Python：数据结构](https://docs.python.org/zh-cn/3/tutorial/datastructures.html) | 列表、字典的基本使用 | 用列表保存数据、字典保存结果 |
| [NumPy 初学者指南（中文译文）](https://numpy.net.cn/doc/stable/user/absolute_beginners.html) | 导入、array、shape、dtype | 创建并打印数组 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 创建最小文件、Conda 环境、安装依赖并记录环境 |
| 1.5h | Python 函数 / 循环练习，实现普通 Python MSE |
| 1h | NumPy 数组与矩阵预测练习 |
| 0.5h | 核对输出，解释函数与数组形状 |
| 1h 机动 | 处理环境、语法和 shape 疑问 |

### 第一步：核对文件并安装依赖

首次从零建立项目时，在根目录创建 `README.md`、`.gitignore`、`scripts/`。当前仓库已包含这些文件、练习脚本和结果；跟做时先阅读现有内容，在草稿或新文件中练习，保留已完成的实验记录。

项目根 `README.md` 专注项目介绍、目标、学习范围和文档导航；本指南写出本周环境配置和逐日运行步骤；实际版本、复现命令索引及结果集中维护在本周实验目录的 README 中。

第一周仅需下面这些直接依赖。以下是最小环境的教学示例，先读懂环境名、channel 与依赖的含义；根目录现有的 [environment.yml](../environment.yml) 已是后续本地 Linux 环境的完整导出快照，包含具体构建、PyTorch / CUDA 包和本机绝对 `prefix`。保留已有快照，首次学习可按下方命令直接创建最小环境。

```yaml
name: petvision
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - numpy
  - matplotlib
  - pip
```

Python 版本由此环境独立管理，NumPy 和 Matplotlib 通过 Conda 安装。pip 预留给后续需要官方 wheel 或 Conda 未提供的包。这里的 channel 配置仅作用于本环境。

核对 `.gitignore` 中是否已有下面的规则；已有文件保留原规则并按需补充：

```gitignore
.conda/
__pycache__/
*.pyc
.env
.cache/
outputs/
```

首次从零建立项目时，按[Git 官方中文入门](https://git-scm.com/book/zh/v2/Git-基础-获取-Git-仓库)准备版本管理：先执行 `git --version`，再用 `git rev-parse --is-inside-work-tree` 检查是否已在仓库中。正常本地目录尚未初始化时执行 `git init`，随后用 `git status --short` 查看状态；已克隆的仓库直接查看状态即可。本步只准备本地记录，远程仓库按需关联。

在已安装 Conda 的终端进入项目根目录。Windows 可使用 Miniforge Prompt / Anaconda Prompt；Linux / WSL / macOS 使用已初始化 Conda 的 shell。先检查：

```bash
conda --version
conda env list
```

首次创建本项目环境：

```bash
conda create -n petvision -c conda-forge --override-channels python=3.12 numpy matplotlib pip
conda activate petvision
```

后续每天学习时执行 `conda activate petvision` 即可。如果已有同名环境，先激活并核对 Python 版本与依赖，创建命令用于首次建立环境。

若当前终端提示需要初始化 Conda，按实际终端选择对应命令，完成后关闭并重新打开终端：

| 终端 | 初始化命令 |
| --- | --- |
| bash | `conda init bash` |
| zsh | `conda init zsh` |
| Windows PowerShell | `conda init powershell` |

激活后检查环境。下面两条导出命令仅用于首次记录；本仓库已有第一周快照，回顾练习时只执行前两条检查命令，依赖变更另存当前周目录，避免覆盖历史环境：

```bash
python --version
python -c "import sys, numpy, matplotlib; print(sys.executable); print(numpy.__version__); print(matplotlib.__version__)"
conda list > experiments/basics/week1/conda-packages.txt
conda list --explicit > experiments/basics/week1/conda-explicit.txt
```

将操作系统、Conda 版本、环境名 `petvision`、Python 版本和解释器路径写入 `experiments/basics/week1/environment.txt`。在编辑器中选择相同的 Conda Python 解释器。

最小 YAML 示例描述直接依赖；根 `environment.yml` 记录当前本地环境的完整导出，受平台和具体构建约束。`conda-packages.txt` 便于查看实际版本，`conda-explicit.txt` 保存同平台 Conda 包的精确清单，不包含 pip 安装包。能说明这些记录的用途即可。第一周快照已经存在；后续周修改依赖时写入对应周目录，保留历史记录。

同操作系统、同架构复现时，可根据显式清单新建一个检查用环境：

```bash
conda create -n petvision-repro --file experiments/basics/week1/conda-explicit.txt
```

跨操作系统或架构时，从上面的最小直接依赖重新解析环境，并另存该平台的快照。若要复现当前完整环境，先在快照副本中移除本机 `prefix`、核对平台构建，再按第二周记录补齐匹配驱动的 PyTorch 包与安装源；现有 YAML 未记录 GPU wheel 的完整安装来源，不能单独保证重建相同 CUDA 构建。第一天完成本机环境与版本记录即可，重建检查可以在 Day 6 机动时间进行。

后续依赖优先通过 Conda 安装；如官方要求 pip，则先激活 `petvision`，再用 `python -m pip` 安装并记录命令与版本。第二周按 [PyTorch 安装说明（中文译文）](https://pytorch.ac.cn/get-started/locally/)（[官方命令向导](https://pytorch.org/get-started/locally/)）选择与笔记本驱动兼容的 GPU 包。

### 第二步：创建 scripts/python_basics.py

先完成三组练习：

1. 创建一个包含五个数字的列表，用循环算出总和与平均值。
2. 创建字典，保存学习日、学习时长、遇到的问题，并打印每个键和值。
3. 实现 `mse(y_true, y_pred)`：逐项求差、平方、累加，再除以样本数。

MSE 即均方误差，本指南定义为所有平方误差的平均值：

```text
MSE = sum((预测值 - 真实值)^2) / 样本数
```

用下面的数据核对第 3 题：

```python
y_true = [1.0, 2.0, 3.0]
y_pred = [1.5, 1.5, 3.0]
```

结果应约为 `0.1666666667`。本题先用普通 Python 的列表与循环完成。再主动制造一次变量名错误，阅读报错位置，修复并解释原因。

### 第三步：创建 scripts/numpy_basics.py

手动输入数据，先在纸上预测形状和结果，再运行核对：

```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
w = np.array([[2.0], [-1.0]])
b = 0.5
y_hat = X @ w + b

print(X.shape, w.shape, y_hat.shape)
print(y_hat)
```

需要解释：

1. X 有几个样本、每个样本几个特征？
2. 为什么 X 与 w 可以相乘？输出形状如何得到？
3. 为什么标量 b 能加到整个输出上？

自查：形状依次为 `(3,2)`、`(2,1)`、`(3,1)`；三个预测值为 `0.5、2.5、4.5`。

### 运行与验收

```bash
python scripts/python_basics.py
python scripts/numpy_basics.py
```

- [ ] 当前环境为 petvision，Python 为 3.12，解释器路径属于该 Conda 环境，两个依赖可导入。
- [ ] 两个脚本能从项目根目录运行，MSE 与矩阵输出正确。
- [ ] 能解释 X、w、y_hat 的 shape。
- [ ] 能说明一次变量名报错的原因，并验证修复后的代码可运行。

<a id="day-2"></a>
## Day 2：掌握数组形状，并写一个小类

### 去哪里学、读到哪里

| 资料 | 阅读范围 |
| --- | --- |
| [NumPy 初学者指南（中文译文）](https://numpy.net.cn/doc/stable/user/absolute_beginners.html) | 索引 / 切片、数组属性、reshape、基本运算、sum / mean |
| [NumPy：广播（中文译文）](https://numpy.net.cn/doc/stable/user/basics.broadcasting.html) | 从末尾维度比较，维度相等或其中之一为 1 时可广播；看基础例子 |
| [Python：类](https://docs.python.org/zh-cn/3/tutorial/classes.html) | 初识类、__init__、实例属性、方法与 self |
| [Python：模块](https://docs.python.org/zh-cn/3/tutorial/modules.html) | import、模块作为脚本运行、__name__ |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 选读数组、广播、类与模块资料 |
| 1.5h | 补充 numpy_basics.py，完成数组操作与错误广播练习 |
| 1h | 创建 linear_model.py，练习类与模块导入 |
| 0.5h | 自查输出，用注释或口头说明形状变化 |
| 1h 机动 | 补语法、转置、axis 或模块导入问题 |

### 第一步：补充 scripts/numpy_basics.py

沿用 Day 1 的 X 与 w，完成下面五项；每项都打印结果和 shape。

1. 取 X 的第一行、第一列、前两行；比较 `X[:, 0]` 与 `X[:, 0:1]` 的维度。
2. 比较 `X.sum(axis=0)` 与 `X.sum(axis=1)`，解释沿哪条轴求和。
3. 比较 `X @ w` 与 `X * w.T`：前者做矩阵乘法，后者做逐元素乘法与广播。
4. 把 w 从 `(2,1)` 变成 `(2,)`，观察预测输出怎样变化。
5. 把 Day 1 的 MSE 改写成 NumPy 版本，与普通 Python 版本对照。

这里 axis 是轴的编号，从 0 开始。二维 X 的 axis=0 对应样本所在的行方向，`sum(axis=0)` 把各行相加，留下每列的和；axis=1 对应列方向，`sum(axis=1)` 把每行内部各列相加，留下每行的和。整数索引 `X[:,0]` 去掉列这一维，切片 `X[:,0:1]` 保留它。

特意检查一种容易静默出错的情况：预测形状为 `(3,1)`，标签却为 `(3,)`。先预测两者相减后的 shape，再运行观察；随后把标签统一为 `(3,1)`。

广播从最右侧对齐维度；对应长度相等或其中一方为 1 时可以扩展。这里 `(3,)` 按 `(1,3)` 对齐，和 `(3,1)` 相减得到 `(3,3)`，相当于每个预测都与全部三个标签比较。我们需要每条样本只对自己的标签求差，所以先用 `y.reshape(-1,1)` 得到 `(3,1)`；`-1` 表示按元素总数自动推算这一维。

在计算 MSE 前检查标签与预测 shape 一致。用注释或口头解释为什么“程序没有报错”仍可能算错损失。

逐项核对输出：`X[:,0]` 为 `(3,)`，`X[:,0:1]` 为 `(3,1)`；`sum(axis=0)` 为 `[9,12]`，`sum(axis=1)` 为 `[3,7,11]`。原 w 为 `(2,1)` 时，`X @ w` 为 `(3,1)`，`X * w.T` 为 `(3,2)`；改为 `(2,)` 后矩阵预测变为 `(3,)`。最后用 Day 1 的三对真实值/预测值比较普通 Python 与 NumPy MSE，二者都应约为 `1/6`，差小于 `1e-10`。

### 第二步：创建 scripts/linear_model.py

实现一个简单的 `LinearModel` 类：

- `__init__(self, w, b)`：保存参数。
- `predict(self, X)`：根据参数返回预测。

在 `numpy_basics.py` 中用 `from linear_model import LinearModel` 导入，验证结果与之前的矩阵公式相同。两个脚本放在同一个 `scripts/` 目录中。

将脚本的演示调用放在 `if __name__ == "__main__":` 下，观察“直接运行”与“被另一个文件导入”的区别。Day 5 的训练脚本采用独立函数，方便 Day 6 检查梯度；本题用于理解状态、方法和模块。

入口保护里的内容包括创建示例数据、实例化模型和打印结果；类和函数定义留在顶层。用同一组 X、w、b 分别调用 `model.predict(X)` 与矩阵公式，比较每个元素，最大绝对差应小于 `1e-10`。下方第三条命令只导入模块，应正常退出且没有演示输出。

### 运行与验收

```bash
python scripts/numpy_basics.py
python scripts/linear_model.py
python -c "import sys; sys.path.insert(0, 'scripts'); import linear_model"
```

- [ ] 能解释 `(N,)` 与 `(N,1)` 的区别，以及转置、reshape、axis 的作用。
- [ ] 能复现错误广播，打印出错误的 `(3,3)` 形状，并正确修复。
- [ ] 类的预测与公式一致，模块导入时演示代码不会自动执行。
- [ ] 能通过代码输出或注释解释形状变化与广播结果。

<a id="day-3"></a>
## Day 3：把矩阵和梯度下降连接起来

### 去哪里学、读到哪里

| 资料 | 阅读范围 |
| --- | --- |
| [D2L 2.3：线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html) | 标量、向量、矩阵、转置、点积、矩阵向量积 |
| [D2L 2.4：微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html) | 导数、变化率与梯度的含义 |

先读公式、图示和解释，数值练习使用 NumPy。书中框架实现可以在第二周对照。

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读标量、向量、矩阵与乘法 |
| 1.5h | 手算形状、点积和线性预测 |
| 0.75h | 阅读导数，手算一次梯度更新 |
| 0.75h | 创建 math_basics.py，核对手算并解释结果 |
| 1h 机动 | 补矩阵乘法或导数的基础疑问 |

### 第一步：在纸上手算形状与线性预测

在纸上列出线性模型中的符号与形状，随后用代码核对：

```text
N：样本数；d：特征数
X：(N,d)
w：(d,1)
b：标量
y_hat = X @ w + b：(N,1)
y：(N,1)
```

手算 Day 1 三条样本的预测，并回答：

1. X 的一行与 w 做点积，得到什么？
2. 为什么 X.T 的 shape 是 `(d,N)`？
3. `X.T @ X` 的 shape 是什么？
4. 特征数从 2 变成 5，w 的形状怎样变化？

<details>
<summary>手算参考：逐条预测与四个问题的解释</summary>

沿用 Day 1 的数据，样本数 `N=3`，每个样本的特征数 `d=2`：

```text
X = [[1, 2],       w = [[ 2],       b = 0.5
     [3, 4],            [-1]]
     [5, 6]]

X.shape = (3,2)
w.shape = (2,1)
b 是标量
```

X 的每一行是一条样本，每一列是一种特征。w 为每种特征分配一个权重：第一个特征乘 2，第二个特征乘 -1；b 是所有样本共用的偏置。`y_hat` 表示模型预测，`y` 表示真实标签，两者在本练习中都采用 `(3,1)` 的形状；真实标签的数值不能由预测直接确定。

**1. 一行与 w 做点积，得到该样本的特征加权和；再加 b 才得到预测。**

对应位置的特征与权重相乘，再将乘积相加：

| 样本 | 点积，即加偏置前的结果 | 加上 b 后的预测 |
| --- | --- | --- |
| 第一行 `[1,2]` | `1×2 + 2×(-1) = 0` | `0 + 0.5 = 0.5` |
| 第二行 `[3,4]` | `3×2 + 4×(-1) = 2` | `2 + 0.5 = 2.5` |
| 第三行 `[5,6]` | `5×2 + 6×(-1) = 4` | `4 + 0.5 = 4.5` |

把三条样本的结果按行放在一起：

```text
X @ w = [[0],      y_hat = X @ w + b = [[0.5],
         [2],                          [2.5],
         [4]]                          [4.5]]

形状：(3,2) @ (2,1) → (3,1)
```

矩阵乘法要求左边的列数等于右边的行数，即两个相邻的维度相等。输出保留左边的行数和右边的列数：`(m,n) @ (n,p) → (m,p)`。这里每条样本得到一个预测，标量 b 通过广播加到每个预测上。

数学上每行的点积是一个数。NumPy 中，`X[0] @ w` 的结果形状为 `(1,)`，`X[0:1] @ w` 为 `(1,1)`；这是索引是否保留行维度带来的区别，两者都只包含第一个样本的加权和 0。

**2. X.T 的形状为 `(d,N)`，因为转置交换矩阵的行与列。**

X 原来的第一列成为转置后的第一行，第二列成为第二行：

```text
X.T = [[1, 3, 5],
       [2, 4, 6]]

形状：(N,d) → (d,N)，本例是 (3,2) → (2,3)
```

**3. `X.T @ X` 的形状为 `(d,d)`，本例是 `(2,2)`。**

先检查形状，再计算每个位置的点积：

```text
(d,N) @ (N,d) → (d,d)
(2,3) @ (3,2) → (2,2)

X.T @ X = [[1×1 + 3×3 + 5×5,  1×2 + 3×4 + 5×6],
           [2×1 + 4×3 + 6×5,  2×2 + 4×4 + 6×6]]

        = [[35, 44],
           [44, 56]]
```

输出中的每个元素，都是 X 的两列对应元素乘积的总和。例如右上角 44 来自第一列 `[1,3,5]` 与第二列 `[2,4,6]` 的点积。

**4. 特征数从 2 变成 5 时，w 需要从 `(2,1)` 变为 `(5,1)`。**

每个特征需要一个权重，因此需要提供五个权重。样本数仍为 3 时，X 变为 `(3,5)`，预测过程为 `(3,5) @ (5,1) → (3,1)`。每条样本仍然输出一个预测，b 仍是标量。

</details>

### 第二步：创建 scripts/math_basics.py

用 NumPy 验证上面的矩阵计算。随后用一个标量函数体验梯度下降：

```text
f(w) = w²
初始 w = 2
学习率 = 0.1
更新规则：w_new = w - 学习率 × f 对 w 的导数
```

先手算一轮：导数为 4，更新后 w=1.6。再写循环执行 10 次，每次打印步数、w 和 f(w)，观察它们是否靠近最小值。

这里“减去梯度”来自局部斜率：w=2 时导数为正，向较小的 w 移动会在足够小的范围内降低 f；w 为负时导数为负，减去它会让 w 增大。学习率决定移动距离，步子过大仍会越过低点甚至发散。因此梯度给出局部方向，不保证任意学习率下每次更新都降低损失。

### 运行与验收

```bash
python scripts/math_basics.py
```

- [ ] 矩阵计算与手算一致，能说明矩阵乘法要求的维度关系。
- [ ] 一维梯度下降第一轮得到 w=1.6，后续 f(w) 持续减小。
- [ ] 能口头或通过代码注释解释：梯度指什么、为什么更新时减去梯度、学习率控制什么。

<a id="day-4"></a>
## Day 4：手推并实现线性回归梯度

### 去哪里学、读到哪里

| 资料 | 阅读范围 |
| --- | --- |
| [D2L 2.4：微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html) | 偏导数、梯度、链式法则 |
| [D2L 3.1：线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html) | 线性模型、损失函数、用梯度下降学习参数 |

本指南的 MSE 是平方误差的平均值。阅读不同资料时留意损失前是否带有 1/2，推导和代码使用同一个定义。

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读偏导、链式法则与线性回归损失 |
| 1.5h | 手推单个样本、多个样本的梯度与形状 |
| 1h | 创建 gradient_basics.py，计算梯度并执行一次更新 |
| 0.5h | 核对公式、形状和 loss，解释更新过程 |
| 1h 机动 | 补推导中不理解的一步 |

### 第一步：在纸上推导 MSE 梯度

从一个样本开始：

```text
y_hat = w1*x1 + w2*x2 + b
L = (y_hat - y)²
```

依次求 L 对 w1、w2、b 的偏导。每次写出“损失对预测的导数 × 预测对参数的导数”，再推广到 N 个样本的平均损失。

继续写出矩阵形式，标注以下形状：X、w、y、误差 E、X.T、dw。db 使用标量。

<details>
<summary>完成手推后展开自查公式</summary>

```text
E = X @ w + b - y             # (N,1)
L = mean(E²)                  # 标量
dw = (2/N) * X.T @ E          # (d,1)
db = (2/N) * sum(E)           # 标量
```

这里 N 是样本数、d 是每条样本的特征数；E 是预测减真实值的误差，L 是所有样本的平均平方误差。dw 表示 L 对各个权重的偏导组成的数组，db 表示 L 对偏置的偏导，二者不是已经更新后的参数。

对一个样本，平方求导得到 `2*(y_hat-y)`，再乘预测对权重的导数 x_j，得到 `2*(y_hat-y)*x_j`。把 N 条样本的贡献求和再除以 N，就得到每个权重的梯度；`X.T @ E` 正是在一次矩阵乘法中完成这些求和。偏置的导数是 1，所以 db 只对误差求和。更新时再使用 `w -= learning_rate*dw`、`b -= learning_rate*db`。

</details>

### 第二步：创建 scripts/gradient_basics.py

按以下接口写函数：

```python
def predict(X, w, b):
    """返回形状为 (N,1) 的预测。"""
    ...

def mse(y_true, y_pred):
    """检查两者 shape 一致，返回平均平方误差。"""
    ...

def gradients(X, y, w, b):
    """返回 dw 和标量 db。"""
    ...
```

使用 Day 1 的 X、标签 `[[1.0], [2.0], [3.0]]`、零初始化 w 与 b，学习率取 0.01。计算更新前的 loss，求出 dw / db，执行一次更新，再重新计算预测和 loss。

本例初始 MSE 为 `14/3`，`dw=[[-44/3],[-56/3]]`、`db=-4`；更新后 w 约为 `[[0.1466666667],[0.1866666667]]`、b 为 `0.04`，重新计算的 MSE 约为 `0.6721185185`。按「更新前 loss → 梯度 → 新参数 → 更新后 loss」打印；若新参数正确而 loss 未变，检查是否仍在使用更新前的预测。

### 运行与验收

```bash
python scripts/gradient_basics.py
```

- [ ] dw 的 shape 与 w 一致，db 是标量，预测与标签均为 `(N,1)`。
- [ ] 使用上述数据与学习率，一次更新后的 loss 小于更新前。
- [ ] 能从单样本链式法则解释矩阵公式，并说明为什么下一轮要重新计算预测。
- [ ] 推导与真实输出一致，关键形状可标在代码注释中。

<a id="day-5"></a>
## Day 5：训练第一份完整模型

### 去哪里学、读到哪里

| 资料 | 阅读范围 |
| --- | --- |
| [D2L 3.1：线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html) | 回顾模型、损失和参数更新的关系 |
| [NumPy：随机数生成器（中文译文）](https://numpy.net.cn/doc/stable/reference/random/generator.html) | default_rng 与 normal 的基本用法 |
| [Matplotlib：折线图（中文译文）](https://matplotlib.net.cn/stable/gallery/lines_bars_and_markers/simple_plot.html) | subplots、plot、坐标轴标签与网格 |
| [pathlib（官方中文）](https://docs.python.org/zh-cn/3.12/library/pathlib.html) | Path、路径拼接与 mkdir |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 查阅随机数和折线图的基本用法 |
| 1h | 创建训练脚本，生成数据，检查预测与梯度 |
| 1.5h | 实现训练循环，打印 loss 与参数 |
| 1h | 保存曲线，核对 loss、参数与训练设置 |
| 1h 机动 | 排查 loss、shape、更新或绘图问题 |

### 第一步：创建文件与目录

- 创建 `scripts/linear_regression_numpy.py`。
- 创建 `experiments/basics/week1/figures/`，用于保存小型实验图表。

### 第二步：固定数据与训练条件

本题使用一个可以核查真实参数的合成任务：

```text
y = 3*x1 - 2*x2 + 1 + 噪声
样本数：200
特征数：2
随机种子：42
X：标准正态分布，shape=(200,2)
噪声：标准差 0.1，shape=(200,1)
真实 w：[[3.0],[-2.0]]，真实 b：1.0
初始 w：全零，shape=(2,1)；初始 b：0.0
训练：全量数据梯度下降，学习率 0.05，更新 500 步
损失：MSE，计算使用浮点数
```

可以用下面的数据生成代码起步，预测、梯度、更新循环由你完成：

```python
import numpy as np

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2))
true_w = np.array([[3.0], [-2.0]])
true_b = 1.0
noise = 0.1 * rng.normal(size=(200, 1))
y = X @ true_w + true_b + noise
```

本周用这些数据核对优化过程，报告训练 MSE 与参数恢复情况；独立训练 / 验证 / 测试划分在第二周正式练习。

本题只有一层线性模型，使用零初始化便于手算。第二周起，多层网络的同层隐藏神经元若全部使用相同的零权重，会难以打破对称性；届时保留 `nn.Linear` 的随机初始化，不把本题的全零权重设置照搬过去。

### 第三步：组织训练脚本

在训练脚本中整理这些函数：

| 函数 | 职责 |
| --- | --- |
| `generate_data(seed=42, n_samples=200)` | 返回 X、y、真实 w、真实 b |
| `predict(X, w, b)` | 返回预测 |
| `mse(y_true, y_pred)` | 返回标量损失 |
| `gradients(X, y, w, b)` | 返回 dw、db |
| `train(X, y, learning_rate, steps)` | 每次调用从零初始化，返回最终 w、b、loss_history |
| `main()` | 调用各步骤、打印结果、保存图表 |

训练时先记录初始 loss，再执行 500 次更新；每次更新后记录重新计算的 loss，因此 loss_history 长度应为 501。每 50 步打印一次进度。

把 main 调用放入 `if __name__ == "__main__":`，为 Day 6 的函数导入做准备。Day 4 的公式和函数可作为重构参考。

### 第四步：保存结果

用 Matplotlib 绘制步数与训练 loss，给坐标轴和标题命名，保存到：

```text
experiments/basics/week1/figures/loss_curve.png
```

可用 `fig.savefig(path, dpi=150, bbox_inches="tight")` 保存，在保存后关闭图。路径从项目根目录计算，文件所在目录先创建。

先创建图和坐标轴，`loss_history` 使用本次 train 返回的损失列表：

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(len(loss_history)), loss_history)
ax.set(xlabel="Update steps", ylabel="Training MSE", title="NumPy linear regression")
ax.grid()
```

再读 pathlib 指定部分，把保存步骤接到这张 fig 后面：

```python
from pathlib import Path

figure_path = Path("experiments/basics/week1/figures/loss_curve.png")
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=150, bbox_inches="tight")
plt.close(fig)
```

横轴取 `0..500`，与 501 个 loss 一一对应。打开实际 PNG 核对标签、曲线和最终数值；相同设置再运行一次，比较初始/最终 loss 和参数，数值差应小于 `1e-10`。

### 运行与验收

```bash
python scripts/linear_regression_numpy.py
```

- [ ] 初始 loss 与最终 loss 都来自实际运行，曲线明显下降。
- [ ] 最终两个权重接近 3 与 -2，偏置接近 1；可将各参数误差小于 0.1 作为本题自查目标。
- [ ] 因为存在噪声，MSE 收敛到较小正数是合理现象。
- [ ] 相同环境、相同种子与相同配置重复运行，得到一致结果。
- [ ] 脚本输出初始 / 最终 loss、最终 w / b、步数与学习率，曲线保存到指定路径。

<a id="day-6"></a>
## Day 6：检查梯度、做对照、重写并复盘

### 去哪里学、读到哪里

| 资料 | 阅读范围 |
| --- | --- |
| [D2L 2.4：微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html) | 回顾导数与有限变化量的关系 |
| [D2L 11.3：梯度下降](https://zh.d2l.ai/chapter_optimization/gd.html) | 一维梯度下降与学习率影响，选读相关图示 |
| [csv（官方中文）](https://docs.python.org/zh-cn/3.12/library/csv.html) | DictWriter、writeheader、writerows、newline |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 阅读梯度检查方法，回顾学习率 |
| 1h | 创建 gradient_check.py，核对全部参数梯度 |
| 1h | 完成三组学习率对照，保存 CSV 和曲线 |
| 1h | 关掉参考，创建 linear_regression_rewrite.py，重写核心流程 |
| 0.5h | 更新实验目录 README 的周总结，核对本周清单 |
| 1h 机动 | 补未通过的检查或重写练习 |

### 第一步：创建 scripts/gradient_check.py

从同目录的 `linear_regression_numpy.py` 导入 generate_data、predict、mse、gradients。使用固定的 5 条样本、非零 w 与 b，数据采用 float64，逐个检查两个权重和一个偏置。

调用 Day 5 的 `generate_data(seed=42, n_samples=5)`，生成固定的 5 条 X、y，参数固定为 `w=[[0.3],[-0.7]]`、`b=0.2`。先复制参数，扰动 `w[0,0]` 时只改这个元素；分别算加/减 epsilon 的 loss 后恢复，再检查另一个权重和 b。打印参数名、解析梯度、数值梯度与绝对差，最后汇总三项最大误差。

数值梯度使用中心差分：

下面 theta 表示本次正在检查的一个标量参数，例如 `w[0,0]`；L 是保持同一份数据、用该参数值重新计算的 MSE。epsilon 是很小的扰动量，不是学习率。解析梯度来自上一天的求导公式，数值梯度来自两次实际 loss 的差，二者应接近。

```text
数值梯度 = [L(theta + epsilon) - L(theta - epsilon)] / (2*epsilon)
epsilon = 1e-6
```

每次只扰动一个参数，保持其他参数不变；计算后恢复原值，再检查下一个参数。将数值梯度与 Day 4 的解析梯度逐项比较，打印差值和最大绝对误差。

在这个小型、float64 的练习中，可把最大绝对误差小于 `1e-5` 作为检查目标。未通过时先检查 MSE 的 1/N、系数 2、参数恢复与标签 shape。

### 第二步：完成学习率对照

创建 `scripts/learning_rate_comparison.py`，从 Day 5 脚本导入 `generate_data` 和 `train`。数据只生成一次，供三组实验共用。

复用 Day 5 数据、初始参数、500 步预算，分别运行学习率：

```text
0.001
0.05
0.5
```

每次 train 调用都重新从零初始化；复用相同数据，避免前一组训练参数影响后一组。

将真实结果写入 `experiments/basics/week1/results.csv`，建议列：

```csv
learning_rate,steps,initial_loss,final_loss,w1,w2,b
```

每组训练结束后，把这些实际数值加入一个字典，三组形成 `rows` 列表，再使用标准库 csv 保存。先读当天资料的 DictWriter 示例：

```python
import csv
from pathlib import Path

result_path = Path("experiments/basics/week1/results.csv")
result_path.parent.mkdir(parents=True, exist_ok=True)
columns = ["learning_rate", "steps", "initial_loss", "final_loss", "w1", "w2", "b"]
with result_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
```

检查 CSV 有一行表头、三行结果，三行 initial_loss 一致，曲线与 CSV 来自同一次训练返回值；不要为填表又随机生成一次数据。后续周的 history 和汇总表沿用这种写法。

将三组曲线放在同一张图，保存为：

```text
experiments/basics/week1/figures/learning_rates.png
```

解释实际观察到的收敛速度与稳定性。机动时间充足时，可在 Day 3 的 w² 例子上比较学习率 0.1 与 1.1，观察过大学习率的影响。

这三组学习率不预先对应“慢、合适、发散”。本仓库的已保存结果中，0.5 比 0.05 更快收敛，两者都稳定；0.001 在 500 步后仍未充分收敛。学习率是否过大取决于损失曲面、特征尺度与当前更新规则，不能只看数值是否大于 0.1。

### 第三步：创建 scripts/linear_regression_rewrite.py

关掉已有训练脚本，保留纸上的公式与函数清单，重新写出：

1. 数据生成与参数初始化。
2. 预测与 MSE。
3. 梯度与更新循环。
4. 初始 / 最终 loss 和参数输出。

按相同条件运行，比较结果。完成后用自己的话解释，仍需依赖参考的部分可在周末小结中简要说明。重写以训练核心流程为主，绘图可复用已有方法。

比较时固定同一份 X、y、零初始化、学习率 0.05 和 500 步，核对两份实现的最终 loss、w、b；相同 float64 条件下，最大绝对差以 `1e-8` 为自查目标。重写脚本自身必须执行并打印这些结果，已有脚本的运行成功不能代替这项检查。

### 第四步：更新 experiments/basics/week1/README.md

周末用约 10–20 分钟更新这份小结，其余时间核对本周验收。集中写清：

1. 本周完成的练习与仍需补齐的部分。
2. 训练命令、配置与关键结果，链接现有曲线和 CSV。
3. 一两条关键理解，例如 MSE 梯度、数组形状或学习率的影响。
4. 遗留问题与典型错误的解决方法，有实际问题时再补充。

公式与推导保留有助于理解的关键步骤即可；用相对链接引用 `figures/loss_curve.png`、`figures/learning_rates.png` 和 `results.csv`，避免重复抄写整份结果。新增运行入口与环境变更补充到本周实验目录的 README。

### 运行与验收

```bash
python scripts/gradient_check.py
python scripts/learning_rate_comparison.py
python scripts/linear_regression_rewrite.py
```

- [ ] 两个权重与偏置的梯度检查全部通过，打印逐项误差和最大绝对误差。
- [ ] 三组学习率共享数据、初始化和步数，CSV 与曲线相互对应。
- [ ] 重写脚本可以运行，能解释每一步的数学与代码。
- [ ] [第一周小结](../experiments/basics/week1/README.md)汇总真实结果与遗留问题，并链接已有曲线和数据表。

## 日常检查与周末整理

平时检查时，提供相关代码和实际输出或图表即可；概念、形状与公式可通过口头说明、纸上推导或简短代码注释解释。

环境快照在依赖变化时更新，实验指标和图表随运行保存。周末集中更新本周实验目录的 README，简要汇总结论与遗留问题即可。

## 本周结束检查

- [ ] 可以运行 NumPy 线性回归，并从零初始化出发学习到接近真实的参数。
- [ ] 可以解释 X、w、b、y、预测、dw 的形状，以及广播可能造成的静默错误。
- [ ] 可以手推 MSE 梯度，使用数值梯度核对实现。
- [ ] 可以在有限参考下重写“预测 → 损失 → 梯度 → 更新”。
- [ ] 保留环境记录、真实曲线和学习率对照，并在实验目录 README 中写出简短周总结。

未通过的基础项先用机动时间补齐，仍未完成则顺延。通过后按照 [StudyPlan.md](StudyPlan.md#stage-1) 进入第二周 PyTorch 与 MNIST。
