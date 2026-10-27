# 第一周：Python、NumPy 与线性回归

学习指南：[第一周逐日任务](../../../docs/week1.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)  
当前本地环境快照：[environment.yml](../../../environment.yml)

本文件集中保存第一周的实际环境、复现命令索引、结果与周总结；首次学习的完整环境准备和操作步骤见上方逐日指南。环境和运行说明随依赖与脚本变化维护；周末花约 10–20 分钟补充关键结果与遗留问题。日常检查直接看代码、输出、图表和概念解释。

## 环境准备

第一周基础环境使用 Conda 管理，名称为 `petvision`，直接依赖是 Python 3.12、NumPy 和 Matplotlib。根目录 `environment.yml` 已更新为包含后续 PyTorch / CUDA 包的本地 Linux 完整快照；第一周实际版本以本目录的历史环境记录为准。

以下命令均在项目根目录执行。先安装 Conda，并使用已初始化 Conda 的终端；终端初始化操作可参阅[第一周环境准备](../../../docs/week1.md#day-1)。

首次创建并激活环境：

```bash
conda create -n petvision -c conda-forge --override-channels python=3.12 numpy matplotlib pip
conda activate petvision
```

已有环境时直接激活：

```bash
conda activate petvision
```

检查解释器和依赖，并在编辑器中选择相同的 Conda Python 解释器：

```bash
python --version
python -c "import sys, numpy, matplotlib; print(sys.executable); print(numpy.__version__); print(matplotlib.__version__)"
```

本周练习使用 CPU。环境有变更时，同步维护依赖定义、安装说明与环境记录。

## 运行练习

先激活 `petvision`，再从项目根目录运行。下表列出第一周已建立的练习入口。训练和学习率对照脚本会自动创建图表输出目录。

| 练习 | 运行命令 |
| --- | --- |
| Python 列表、字典与 MSE | `python scripts/python_basics.py` |
| NumPy 数组、矩阵乘法与广播 | `python scripts/numpy_basics.py` |
| 类与模块 | `python scripts/linear_model.py` |
| 矩阵与一维梯度下降 | `python scripts/math_basics.py` |
| MSE 梯度与单步更新 | `python scripts/gradient_basics.py` |
| NumPy 线性回归训练 | `python scripts/linear_regression_numpy.py` |
| 数值梯度检查 | `python scripts/gradient_check.py` |
| 学习率对照 | `python scripts/learning_rate_comparison.py` |
| 有限参考重写 | `python scripts/linear_regression_rewrite.py` |

输入设置、公式、自查结果和验收标准见[逐日指南](../../../docs/week1.md)。新增脚本的运行方法在本表补充。

## 周总结

本周练习从 Python、NumPy 基础推进到线性回归训练、数值梯度检查和学习率对照。以下总结依据现有代码、实际运行结果与保存的实验文件。

### 完成情况与关键理解

- Python 练习覆盖列表、字典、循环、函数、类、模块导入及脚本入口保护；NumPy 练习覆盖索引、切片、转置、广播、矩阵乘法和沿轴求和。
- 完成线性预测、MSE、解析梯度、参数更新和损失曲线保存；训练记录包含初始损失与 500 次更新后的损失，共 501 个点。
- 完成两个权重与偏置的数值梯度检查，以及三组学习率在相同条件下的对照实验。
- 重写脚本中已定义 `predict`、`mse`、`gradients`、数据生成和训练循环；在相同配置下，训练结果与基准脚本一致。

线性回归使用以下公式，MSE 定义不带 `1/2`：

```text
y_hat = X @ w + b            # (N,1)
E = y_hat - y                # (N,1)
L = mean(E²)                 # 标量
dw = (2/N) * X.T @ E         # (d,1)
db = (2/N) * sum(E)          # 标量
w_new = w - learning_rate * dw
b_new = b - learning_rate * db
```

其中 N 是样本数、d 是每条样本的特征数；`X` 为 `(N,d)`，`w` 为 `(d,1)`，`b` 为标量，标签 `y` 与预测均为 `(N,1)`。系数 2 来自平方求导，除以 N 来自求平均，`X.T @ E` 为每个特征汇总所有样本的梯度贡献。参数更新后，需要重新计算预测和损失。

数值梯度通过参数两侧的损失变化估算斜率，与解析梯度比较；它用于核对公式实现。`main()` 组织执行流程，入口保护确保导入模块时不会自动启动训练。

### 实验摘要

- 训练命令见上方运行入口。学习率对照固定随机种子 `42`、样本数 `200`，三组共用相同数据，每组从零初始化并更新 `500` 步。
- 基准配置（学习率 `0.05`）的初始 / 最终 loss：`12.7209132622 → 0.0102533295`；最终 `w≈[2.998216,-2.005331]`、`b≈0.993054`。
- 数值梯度检查：最大绝对误差为 `6.396252e-10`，小于阈值 `1e-5`，两个权重和偏置均通过；逐项误差由脚本打印。
- 三组学习率的主要差异：`0.001` 在 500 步后 loss 仍约为 `2.06994`，尚未充分收敛；`0.05` 和 `0.5` 都稳定收敛到约 `0.0102533`。本次数据上，二者首次达到 loss 不超过 `0.011` 分别需要 54 步和 3 步，`0.5` 收敛更快。
- 重写训练：与基准脚本使用相同数据、初始化、学习率和步数，501 项损失记录及最终参数均一致。数据含有标准差为 `0.1` 的噪声，本实验最终 MSE 为较小正数。

梯度检查配置：随机种子 `42`、样本数 `5`、数据类型 `float64`、检查位置 `w=[[0.3],[-0.7]]` 与 `b=0.2`，中心差分步长 `epsilon=1e-6`。脚本比较固定参数处的解析梯度与数值梯度。

逐组参数和损失见[学习率对照结果](results.csv)，曲线见[学习率对照图](figures/learning_rates.png)、[基准训练曲线](figures/loss_curve.png)和[重写训练曲线](figures/loss_curve2.png)。学习率对照图使用对数纵轴，便于同时观察初期下降与接近收敛时的差异。

### 遗留问题

本次核对中，重写脚本的三个核心函数已在文件内定义，学习率对照 CSV 与三张曲线均已保存，相关结果链接有效。

本周出现并处理过的问题：

- 标签为 `(N,)`、预测为 `(N,1)` 时，相减会广播成 `(N,N)`；在 MSE 与梯度函数中检查形状一致，避免静默计算错误。
- 更新次数与日志编号容易错位；初始损失记为第 0 步，循环中实际更新次数使用 `step + 1`，每 50 次更新打印一次。
- 输出目录不存在时保存图片会失败；保存前用 `mkdir(parents=True, exist_ok=True)` 创建目录，保存后关闭图。

后续继续巩固：在有限参考下复述链式法则到矩阵梯度的推导，并独立重写预测、损失、梯度和更新循环。当前实验报告训练 MSE；独立训练、验证和测试划分按计划在第二周练习。

## 环境与复现材料

- [实际环境](environment.txt)：系统、Conda 版本、环境名、Python 版本和解释器路径。
- [包版本清单](conda-packages.txt)：实际安装包、版本和 channel。
- [同平台精确清单](conda-explicit.txt)：Conda 包的精确清单，用于同操作系统、同架构重建。
- 根目录 `environment.yml`：当前本地 Linux 环境完整快照，包含平台构建和本机路径；与第一周历史环境的阶段不同。

首次完成第一周安装时，在激活的环境中导出包清单；本目录已有快照，后续依赖变更写入对应周目录，保留第一周记录：

```bash
conda list > experiments/basics/week1/conda-packages.txt
conda list --explicit > experiments/basics/week1/conda-explicit.txt
```

同操作系统、同架构复现时，可新建一个检查用环境：

```bash
conda create -n petvision-repro --file experiments/basics/week1/conda-explicit.txt
conda activate petvision-repro
```

跨平台时，按第一周指南的最小直接依赖重新创建环境并另存快照。显式清单只恢复 Conda 包；后续用 pip 补充依赖时，将完整安装命令（含包索引地址）与版本清单保存在对应周说明中。
