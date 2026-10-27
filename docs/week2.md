# 第二周逐日学习指南：PyTorch、MNIST MLP 与完整训练流程

> 更新日期：2026-10-27<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-1)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：用 autograd 对照手写梯度，完成 MNIST 数据划分、MLP 训练、小批过拟合、最佳 checkpoint、独立测试与加载一致性检查。  
> 环境：本地 Conda 环境 `petvision`，Python 3.12；按官方安装向导补充匹配驱动的 PyTorch / torchvision，显存以实际检查为准。

## 怎么使用这份指南

先确认[上一周指南](week1.md)的基础关卡已通过。应能解释第一周的 MSE 梯度和 `(N,d) @ (d,1)`；本周将手写梯度对应到自动求导，并把回归任务扩展为分类。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/mnist_mlp/week2/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明计划创建或修改时间。当前已有 Day1–4 的数据、模型、训练 / 验证、配置入口和日志绘图；Day5 的 checkpoint 与独立评估入口已接通，现有 6 项测试通过。正式五轮训练的配置、日志、曲线和 best/last 已生成；best 为第 5 轮，独立加载后完整验证集 accuracy=96.42%，与该轮日志一致。Day4 未配套权重的记录已归档，接下来按 Day6 做最终测试。已有结果与检查范围见[第二周记录](../experiments/mnist_mlp/week2/README.md)，下面的目录图描述本周完整目标。

```text
PetVisionLab/
├── pyproject.toml                         # Day 2：最小 src 包配置
├── environment.yml                        # Day 1：补充实际依赖
├── configs/cv/mlp_mnist.yaml               # Day 4：训练配置
├── scripts/
│   ├── check_environment.py               # Day 1：设备、版本与显存检查
│   ├── torch_basics.py                    # Day 1：Tensor、autograd 对照
│   ├── inspect_mnist.py                   # Day 2：数据与划分检查
│   ├── check_mnist_training.py            # Day 4 第三步：独立 smoke 与 64 样本拟合
│   ├── train_cv.py                        # Day 4：训练、smoke、过拟合入口
│   └── evaluate.py                        # Day 5：加载 checkpoint 评估
├── src/petvision/
│   ├── __init__.py                        # Day 2：各新增包目录补 __init__.py
│   ├── data/datasets.py                   # Day 2：加载 / 划分；Day 4：参数化接口
│   ├── models/cv/mlp.py                   # Day 3：固定 MLP；Day 4：参数化构造
│   ├── training/cv_trainer.py             # Day 4：train_one_epoch / evaluate
│   └── utils/{seed,checkpoint}.py         # Day 4–5：种子与保存加载
├── tests/test_cv_training.py              # Day 3–6：形状、梯度、加载检查
├── experiments/mnist_mlp/week2/
│   ├── README.md                         # 已提供：环境、运行与周总结
│   ├── mnist_split.json                  # Day 2：固定索引
│   ├── history.csv                       # Day 5：每个 epoch 的真实指标
│   └── figures/training_curves.png        # Day 5：训练/验证曲线
└── outputs/mnist_mlp/{best,last}.pt        # Day 5：实际训练生成，不上传
```

## 固定条件与接口约定

固定种子 42。MNIST 官方训练集划分为 55,000 / 5,000，官方 10,000 条测试集保留到 Day 6。输入 `[B,1,28,28]`，标签 `[B]`、`torch.long`；MLP 为 `784→256→128→10`，中间 ReLU，返回 logits。起始配置：batch=128、Adam、lr=1e-3、5 epochs、FP32、num_workers=0，预处理 ToTensor 后使用固定 mean=0.5、std=0.5。先用 CPU 和小批检查，再正式使用 GPU。

新增源码按 `src/petvision/` 布局；用最小 `pyproject.toml` 安装本地包后统一 `from petvision...` 导入。配置用 YAML；读取只需 `yaml.safe_load`，先写普通函数，不增加复杂 Trainer 框架。`train_cv.py` 接收 `--config`、`--smoke`、`--overfit-samples`；`evaluate.py` 接收 `--config`、`--checkpoint`、`--split val|test`。配置字段至少含 model、data、training、experiment_dir、output_dir。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | Tensor、自动求导与 GPU 环境 | 4h | 1h | 环境检查、手写梯度与 autograd 一致 |
| Day 2 | 数据加载与独立划分 | 4h | 1h | MNIST 55k/5k 索引与 DataLoader |
| Day 3 | MLP、logits 与交叉熵 | 4h | 1h | 三层 MLP、分类损失与反向传播 |
| Day 4 | 训练循环与小批过拟合 | 4h | 1h | 参数化模型 / 数据接口、配置接线与小样本检查 |
| Day 5 | 最佳 checkpoint 与加载一致性 | 4h | 1h | best/last 保存及独立评估入口 |
| Day 6 | 配置冻结、独立测试与周总结 | 4h | 1h | MNIST 测试结果、误分类与复现说明 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：Tensor、自动求导与 GPU 环境

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 安装说明（中文译文）](https://pytorch.ac.cn/get-started/locally/)（[官方命令向导](https://pytorch.org/get-started/locally/)） | 按系统、驱动与计算平台选择稳定安装命令 | 在 Conda 环境内安装匹配的 torch / torchvision |
| [Tensor 与 autograd（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html) | 计算梯度、禁用梯度跟踪、计算图；末尾可选阅读只看梯度累加示例，Jacobian 推导可跳过 | 用 NumPy 对照 backward，再观察累加与清零 |
| [Tensor 基础（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/tensorqs_tutorial.html) | 创建、属性、reshape、与 NumPy 的桥接 | 打印 shape、dtype、device，并完成转换 |
| Tensor API 速查（中文译文）：[clone](https://pytorch.ac.cn/docs/stable/generated/torch.clone.html)、[detach](https://pytorch.ac.cn/docs/stable/generated/torch.Tensor.detach.html)、[is_leaf](https://pytorch.ac.cn/docs/stable/generated/torch.Tensor.is_leaf.html)、[isfinite](https://pytorch.ac.cn/docs/stable/generated/torch.isfinite.html) | 按下面第三至第五步用到的地方查阅；区分复制数值、断开计算图、叶子属性和有限值检查 | 保存首次梯度副本，检查 Tensor 属性与梯度 |
| [argparse 入门（官方中文）](https://docs.python.org/zh-cn/3.12/howto/argparse.html) | 概念、介绍可选参数；只看 ArgumentParser、add_argument、parse_args、type 与 --help | 让脚本接收 --output-dir |
| [pathlib（官方中文）](https://docs.python.org/zh-cn/3.12/library/pathlib.html) | 基础使用，以及 Path.mkdir、Path.write_text | 拼接路径、创建目录、写入 UTF-8 文本 |
| [Python 模块（官方中文）](https://docs.python.org/zh-cn/3.12/tutorial/modules.html#executing-modules-as-scripts) | 复习 6.1.1「以脚本方式执行模块」中的 __name__ 与入口保护 | 让直接运行执行练习，导入只提供定义 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 检查环境与官方安装方式，完成设备验证 |
| 1.5h | 创建 Tensor，完成零参数、非零参数的 NumPy / autograd 对照 |
| 0.5h | 练习命令行参数、路径与环境记录写入 |
| 1h | 验证梯度累加、清零、detach 与入口保护，核对当天验收项 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 按下面第一步激活环境、补齐 Conda 工具依赖、安装兼容的 PyTorch / torchvision，再验证 GPU；记录实际命令、版本、驱动、GPU 型号、总显存与解释器路径。

2. 创建 `scripts/check_environment.py`：打印 Python、torch、torchvision、CUDA 是否可用、设备名及 `torch.cuda.mem_get_info()`；先按下面第二步练习参数解析和文本写入，再接入实际检查信息。提供 `--output-dir` 时保存本周 `environment.txt`，不提供时只打印；CPU 路径也能完成检查。

3. 创建 `scripts/torch_basics.py`，按第三步分别完成零参数与非零参数的 MSE、NumPy 解析梯度和 autograd 梯度对照；两组都打印 loss 与梯度误差，误差目标 `1e-8`。

4. 按第四步保留同一组非零参数，每次重新计算 loss 后再 backward，观察「单次 → 累加两次 → 清零后单次」三个阶段；期间保持参数数值不变。

5. 按第五步打印 Tensor 属性、比较 detach 前后的梯度记录、转换为 NumPy，并检查梯度是否有限。

6. 按第六步将演示放进 `main()`，添加入口保护；用直接运行和导入两种方式验证执行范围。

### 分步操作与自查

<a id="environment"></a>
#### 第一步：在 petvision 中安装并验证 PyTorch

在项目根目录打开终端，先沿用第一周环境，再补充本周使用的 YAML 读取与检查工具：

```bash
conda activate petvision
python --version
python -c "import sys; print(sys.executable)"
conda install -c conda-forge pyyaml pytest
```

Python 应为 3.12，解释器路径应属于 `petvision`。PyYAML 用于读取第四天的训练配置，pytest 用于执行第三天开始编写的检查。首次安装时，Conda 基础依赖准备好后，再安装下面的 PyTorch wheel；wheel 是已经编译好的安装包。若沿用已经装好 pip 包的环境，先检查现有依赖；需要改变 Conda 底层依赖时，在新环境验证并重装配套 pip 包，保留原环境和快照，参见总计划的环境管理约定。

先在当前终端查看驱动是否能识别显卡：

```bash
nvidia-smi
```

保存 GPU 型号、Driver Version 和显存总量。第一周环境记录是 WSL2；若继续在 WSL2 学习，PyTorch 安装向导的操作系统选择 **Linux**。WSL 使用 Windows 主机提供的 NVIDIA 驱动，驱动问题应先检查主机与 WSL 的支持情况，具体见 [NVIDIA WSL 说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

打开 [PyTorch 安装说明（中文译文）](https://pytorch.ac.cn/get-started/locally/)（[官方命令向导](https://pytorch.org/get-started/locally/)），逐项选择：

| 选项 | 本项目如何选择 |
| --- | --- |
| PyTorch Build | Stable |
| Your OS | 当前 Python 所在系统；WSL2 选 Linux，原生 Windows 选 Windows |
| Package | Pip，在已激活的 Conda 环境中安装 |
| Language | Python |
| Compute Platform | 与实际 NVIDIA 驱动兼容的 CUDA 构建；临时只做 CPU 检查时选择 CPU |

将页面生成命令开头的 `pip` 或 `pip3` 改成 `python -m pip`，其余包名、版本与 `--index-url` 保持该命令的配套组合，再执行。torch 与 torchvision 一起按向导安装，避免分别选择互不兼容的版本。把执行过的完整命令记入本周实验记录；后续复现使用这次验证过的组合。

`nvidia-smi` 中的 CUDA Version 表示驱动支持能力，`torch.version.cuda` 表示当前 PyTorch 构建对应的 CUDA 版本，两者不要求完全相同。选择依据是驱动与所选构建的兼容关系，可查 [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。本周使用预编译 PyTorch 包，不编译 CUDA 扩展，先验证安装包能否工作，不以额外安装完整 Toolkit 作为前置任务。

安装后检查依赖：

```bash
python -m pip check
python -c "import torch, torchvision, yaml, pytest; print(torch.__version__); print(torchvision.__version__)"
```

先用下面片段核对实际环境；再按第二步学习 `argparse` 与 `pathlib`，把检查整理进 `scripts/check_environment.py`，让终端与 `environment.txt` 使用同一份信息：

```python
import sys
import torch
import torchvision

print(sys.executable)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM total/free: {total / 1024**3:.2f}/{free / 1024**3:.2f} GiB")
    x = torch.ones((2, 2), device="cuda")
    print("GPU matrix product:", x @ x)
```

矩阵乘积应为一个各元素都是 2 的 2×2 Tensor，设备应为 CUDA。能够打印显卡名称和能够执行一次 GPU 运算都要检查；本周梯度数值对照仍先使用 CPU float64。

若安装或 CUDA 检查失败，按下列顺序定位：

| 现象 | 先检查什么 | 如何处理 |
| --- | --- | --- |
| `No module named torch` | `sys.executable`、`python -m pip --version` 是否属于 petvision | 重新激活环境，并在编辑器中选择同一个解释器，再安装 |
| torch 能导入，`torch.version.cuda` 为 None | 当前是否装了 CPU 构建 | 若要使用 NVIDIA GPU，按官方向导选择兼容 CUDA 构建重新安装配套包 |
| CUDA 构建已安装，但 `is_available()` 为 False | `nvidia-smi` 是否正常、驱动是否满足构建要求 | 先解决驱动或 WSL 的 GPU 访问，再检查 PyTorch 包；WSL 内不另装 Linux 显卡驱动 |
| torchvision 导入时报算子或动态库错误 | torch / torchvision 是否来自同一次兼容安装 | 核对版本和来源，按向导安装配套版本，不靠反复升级单个包排错 |
| 显卡可见，但 GPU 运算失败或显存不足 | 完整报错、剩余显存、是否有其他占用 | 先用上面的 2×2 运算定位；正式训练再按实际情况减小 batch |

GPU 问题暂未解决时，先用 CPU 完成当天 Tensor 与梯度练习，并在环境记录中保留问题。将 `pyyaml`、`pytest` 等实际新增依赖同步到根 `environment.yml`；PyTorch 的版本与安装源也要保留。安装验证后导出：

```bash
conda list > experiments/mnist_mlp/week2/conda-packages.txt
conda list --explicit > experiments/mnist_mlp/week2/conda-explicit.txt
python -m pip freeze > experiments/mnist_mlp/week2/pip-freeze.txt
```

Conda explicit 清单记录 Conda 包；pip 补充安装还需相应版本清单和原安装命令。依赖不变时，不必每天重新导出。

<a id="environment-output"></a>
#### 第二步：让脚本接收输出目录并保存环境信息

先读当天资料中的 argparse 入门和 pathlib 指定部分。命令行参数用来把终端输入传给脚本；例如运行 `python scripts/check_environment.py --output-dir experiments/mnist_mlp/week2` 时，脚本需要读到后面的目录路径。

先用下面这个只依赖标准库的最小例子理解流程，再把第一步的实际 PyTorch / GPU 信息加入其中：

```python
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="检查并记录运行环境")
    parser.add_argument("--output-dir", type=Path, help="环境记录的输出目录")
    args = parser.parse_args()

    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Interpreter: {sys.executable}",
    ]
    report_text = "\n".join(lines) + "\n"
    print(report_text, end="")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "environment.txt"
        output_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
```

按数据流解释每一步：

1. `ArgumentParser()` 创建参数解析器，`add_argument` 声明脚本接受的参数，`parse_args()` 读取实际命令。参数名中的连字符会转换成属性名中的下划线，因此使用 `args.output_dir`。
2. `type=Path` 把输入的路径字符串转换为 Path 对象；转换本身不创建目录。不提供这个可选参数时，值默认为 None，本例只打印。
3. `"\n".join(lines)` 把每条信息拼成多行文本；打印与文件写入都使用 `report_text`，避免分别收集两次造成遗漏或口径不同。
4. `args.output_dir / "environment.txt"` 使用 `/` 拼接路径。`mkdir` 的 `parents=True` 会补齐缺少的父目录，`exist_ok=True` 允许目录已经存在；`write_text` 按指定编码写入文本，重复运行会更新同名文件。

随后将 torch / torchvision 版本、`torch.version.cuda`、CUDA 可用性及可用时的设备名和显存逐行加入 `lines`，保留 CPU 分支；这些值都来自实际检查。可以用 f-string 先格式化显存，再放入列表。GPU 小矩阵运算用于验证设备能够工作，其检查逻辑也保留。

用 `python scripts/check_environment.py --help` 查看参数说明，再分别尝试不带参数和带 `--output-dir` 的运行方式。打开生成的 `environment.txt`，确认环境字段与终端打印一致；再次运行应能够更新记录。后面的 Day 4 在此基础上增加训练配置与检查模式的参数。

<a id="torch-numpy-check"></a>
#### 第三步：在脚本里独立比较 NumPy 与 autograd

下面第三至第六步都写在 `scripts/torch_basics.py` 中。先完成各段计算，最后按第六步的结构整理到 `main()`；解释可以写成代码注释或口头复盘。

沿用第一周数据：X=`[[1,2],[3,4],[5,6]]`、y=`[[1],[2],[3]]`。今天的数值对照统一使用 **CPU、float64**，保持 y 为 `(3,1)`，避免与预测相减时错误广播。

先做零参数例子，理解一次前向与反向：

```python
X = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
y = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
w = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
loss = ((X @ w + b - y) ** 2).mean()
loss.backward()
print("loss:", loss.item())
print("autograd dw:", w.grad)
print("autograd db:", b.grad)
```

随后完成下面两组对照；每组都从新建的 w、b 开始，各做一次 backward：

| 参数 | MSE 参考值 | dw 参考值 | db 参考值 |
| --- | --- | --- | --- |
| `w=[[0.0],[0.0]]`，`b=0.0` | `14/3`，约 `4.6666666667` | `[[-44/3],[-56/3]]` | `-4.0` |
| `w=[[0.3],[-0.7]]`，`b=0.2` | `15.85` | `[[-27.0],[-34.4]]` | `-7.4` |

每组按相同顺序操作：

1. 用 NumPy 为相同的 X、y、w、b 建立 float64 数组与标量，命名为 `X_np、y_np、w_np、b_np`；在文件顶部添加 `import numpy as np`。
2. 按第一周公式独立计算损失与解析梯度。也可调用第一周 `gradient_basics.py` 中已验证的 `predict、mse、gradients`；直接运行 `scripts/torch_basics.py` 时，同目录下可用 `from gradient_basics import ...` 导入。

   ```python
   residual_np = X_np @ w_np + b_np - y_np
   loss_np = np.mean(residual_np ** 2)
   dw_np = (2 / X_np.shape[0]) * X_np.T @ residual_np
   db_np = (2 / X_np.shape[0]) * np.sum(residual_np)
   ```

3. 用相同数值新建 PyTorch 的 w、b，设置 `requires_grad=True`，计算 loss 并调用一次 backward。X、y 本题不需要梯度。
4. 将 autograd 结果与 NumPy 结果比较。这里提前用到的 `detach().cpu().numpy()` 会在第五步展开；标量可用 `.item()` 取出数值。

   ```python
   loss_error = abs(loss.item() - loss_np)
   dw_error = np.max(np.abs(w.grad.detach().cpu().numpy() - dw_np))
   db_error = abs(b.grad.item() - db_np)
   print("loss error:", loss_error)
   print("dw max abs error:", dw_error)
   print("db abs error:", db_error)
   assert loss_error < 1e-8
   assert dw_error < 1e-8
   assert db_error < 1e-8
   ```

两组都应打印 MSE、NumPy 解析梯度、autograd 梯度和误差。上表用于人工核对；脚本中的解析梯度仍由 NumPy 公式实际算出。只打印 `w.grad、b.grad`，还没有完成独立对照。

<a id="torch-gradient-accumulation"></a>
#### 第四步：用同一组参数验证梯度累加与清零

接着第三步的**非零参数**例子继续做。此时已经完成一次 backward，`w.grad` 应为 `[[-27.0],[-34.4]]`，`b.grad` 应为 `-7.4`。

这一小节保持 X、y 和参数值不变，始终使用同一组 w、b；重新创建参数或更新参数数值都会干扰本题的累加对照。重新计算 loss 会建立新的计算图，但不会自动清除参数上已有的 `.grad`。

1. 保存首次梯度的副本，用于后续比较：

   ```python
   first_dw = w.grad.clone()
   first_db = b.grad.clone()
   ```

   `.clone()` 复制当前数值。`first_dw = w.grad` 只是让两个名字指向同一个 Tensor，后续累加可能使「首次记录」也跟着改变。

2. **保留当前梯度**，重新前向计算，再做第二次 backward：

   ```python
   loss2 = ((X @ w + b - y) ** 2).mean()
   loss2.backward()
   ```

   打印此时的梯度，检查它们分别等于 `2 * first_dw`、`2 * first_db`，误差小于 `1e-8`。例如 w 的比较可写为 `assert (w.grad - 2 * first_dw).abs().max().item() < 1e-8`；自行补齐 b 的检查。

3. 清除已有梯度，并打印清除后的状态：

   ```python
   w.grad = None
   b.grad = None
   print("after reset:", w.grad, b.grad)
   ```

   此时应为 `None、None`。这里清除的是梯度记录，w、b 的参数值仍保持原样。

4. 再重新计算一份 `loss3` 并 backward，检查新梯度分别等于 `first_dw、first_db`，误差小于 `1e-8`。每次反向后都打印阶段名称、对应的 MSE 和梯度，完整展示：

   | 阶段 | w.grad | b.grad |
   | --- | --- | --- |
   | 第一次 backward | `[[-27.0],[-34.4]]` | `-7.4` |
   | 保留梯度，再前向与 backward | `[[-54.0],[-68.8]]` | `-14.8` |
   | 清除梯度，再前向与 backward | `[[-27.0],[-34.4]]` | `-7.4` |

每次 backward 前都重新前向建图。对同一份已经反向过的 loss 直接再次调用 backward，通常会遇到计算图已释放的报错。本题通过重新计算 loss 练习常见训练流程，不需要设置 `retain_graph=True`。虽然参数上的梯度发生了累加，参数值没有更新，因此三个阶段的 MSE 都应仍为 `15.85`。

<a id="torch-detach-check"></a>
#### 第五步：打印 Tensor 属性，观察 detach 与 NumPy 转换

先打印 X、y、w、b 的 `shape、dtype、device、requires_grad、is_leaf`，对照本题的预期：

| Tensor | shape | dtype / device | requires_grad | is_leaf |
| --- | --- | --- | --- | --- |
| X | `(3,2)` | float64 / cpu | False | True |
| y | `(3,1)` | float64 / cpu | False | True |
| w | `(2,1)` | float64 / cpu | True | True |
| b | `()`，零维标量 | float64 / cpu | True | True |

本题的 w、b 是直接创建的叶子参数，默认会在 backward 后保存 `.grad`；`w.grad.shape` 应与 `w.shape` 相同，b 也一样。预测 `X @ w + b` 由运算得到，`requires_grad=True`、`is_leaf=False`。叶子属性与是否需要梯度是两个属性；本题的 X、y 虽然不需要梯度，也属于叶子 Tensor。

再用预测值观察 detach：

```python
prediction = X @ w + b
detached_prediction = prediction.detach()
print("prediction requires_grad:", prediction.requires_grad)
print("detached requires_grad:", detached_prediction.requires_grad)
print("original still requires_grad:", prediction.requires_grad)
prediction_np = detached_prediction.cpu().numpy()
print("NumPy prediction:", prediction_np, prediction_np.shape)
```

三个布尔值应依次为 `True、False、True`，转换后的 NumPy 形状为 `(3,1)`。detach 返回脱离原计算图的 Tensor，原 prediction 仍保留梯度跟踪；它也不会清除 w、b 已有的梯度。detach 与原 Tensor 共享存储，此处只读取数值；保存首次梯度快照仍使用第四步的 clone。

将需要梯度的 Tensor 转为 NumPy 时，使用 `tensor.detach().cpu().numpy()`：detach 断开计算图，cpu 确保数据位于 CPU，numpy 再转换为数组。今天虽已在 CPU 上，仍练习这条完整路径；第三步的 NumPy 对照就是它的一种用途。

在第四步最后一次 backward 后，再补上有限值检查，确认梯度没有 NaN 或 Inf：

```python
assert w.grad is not None and b.grad is not None
assert torch.isfinite(w.grad).all().item()
assert torch.isfinite(b.grad).all().item()
```

能解释两个问题：为什么本题只为 w、b 启用梯度？为什么 `detach()` 与将 `.grad` 设为 None 分别用于不同操作？

<a id="torch-entry"></a>
#### 第六步：加入入口保护，分别验证直接运行与导入

将数据创建、前向、backward、打印和断言放进 `main()`；导入语句和函数定义放在文件顶层。文件结构如下，`pass` 用你已完成的各段练习替换：

```python
import numpy as np
import torch


def main():
    # 依次完成：两组 NumPy 对照、梯度累加与清零、属性与 detach 检查。
    pass


if __name__ == "__main__":
    main()
```

直接运行脚本时，`__name__` 为 `"__main__"`，会调用 main。被其他代码导入时，入口判断不成立，只加载导入语句和函数定义，练习不会自动运行。若在入口保护外仍留着创建数据、backward 或 print，这些语句在导入时仍会执行。

用下面的第三条命令验证导入。`sys.path.insert(0, 'scripts')` 只为这次命令临时加入脚本搜索目录，便于从项目根目录导入 `torch_basics` 及其同目录模块；本周 Day 2 再建立正式的 src 包。

### 运行与验收

完成两个脚本后，在已激活环境的项目根目录执行：

```bash
python scripts/check_environment.py --output-dir experiments/mnist_mlp/week2
python scripts/torch_basics.py
python -c "import sys; sys.path.insert(0, 'scripts'); import torch_basics"
```

第一条打印并保存实际环境；第二条输出下列学习结果；第三条应正常退出，且没有练习输出。运行成功后，逐项核对：

- [ ] **环境与参数**：能解释 --output-dir 如何变成 Path，理解目录创建与文本写入；environment.txt 与终端的环境字段一致，不提供参数时只打印，CPU 分支也能完成检查。
- [ ] **NumPy 独立对照**：零参数与非零参数两组都打印 MSE、NumPy 解析梯度、autograd 梯度；loss、dw、db 的误差均小于 `1e-8`。
- [ ] **梯度累加与清零**：同一组非零参数展示三阶段梯度，验证「单次 → 两倍 → 单次」；每次反向前重新前向，参数值与 MSE 保持不变。
- [ ] **Tensor 属性与有限值**：打印并解释 shape、dtype、device、requires_grad、is_leaf；参数与梯度同形状，梯度没有 NaN 或 Inf。
- [ ] **detach 与 NumPy**：展示原 Tensor 和 detach 结果的 requires_grad 不同，原 Tensor 属性保持不变；能转换为 NumPy，能区分 detach 与清除梯度。
- [ ] **入口保护**：直接运行执行全部练习，导入只加载定义，没有练习输出或自动 backward。
- [ ] **设备记录**：GPU 可用性和显存有真实记录；安装问题未解决时先完成 CPU 小样本检查。实际安装命令、版本和驱动记录保存到本周实验 README，周末集中整理。

<a id="day-2"></a>
## Day 2：MNIST 数据加载与固定划分

今天处理的数据集是 **MNIST 手写数字数据集**：图片为 28×28 的单通道灰度图，标签是 0–9。今天要完成的顺序是：下载并读取 MNIST → 固定训练 / 验证划分 → 组成数据批次 → 检查图片和标签。第一步的包配置是为了让这些数据处理代码能被后续训练脚本复用。

三份数据的职责不同，今天先准备好它们，后续再训练和评估：

| 名称 | 来源与数量 | 后续用途 |
| --- | --- | --- |
| train，训练集 | 官方训练集中的 55,000 张 | 计算训练损失，通过反向传播更新模型参数 |
| val，验证集 | 官方训练集中的另外 5,000 张 | 比较配置、选择最佳 checkpoint；不用于反向传播 |
| test，测试集 | 官方独立测试集的 10,000 张 | 模型和配置确定后评估最终效果，不根据它调参 |

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Dataset / DataLoader（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/data_tutorial.html) | Dataset、DataLoader、批次与可视化 | 检查第一批图片与标签 |
| [MNIST 数据集接口（torchvision 官方英文）](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) | root、train、download、transform，以及单样本的返回值 | 下载 MNIST，区分官方训练集与官方测试集 |
| [PyTorch 基础教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) | 数据变换与模块组织相关小节 | 组织最小 src 包 |
| [Python：包（官方中文）](https://docs.python.org/zh-cn/3.12/tutorial/modules.html#packages) | 6.4 的目录示例、__init__.py 与点号导入；先不展开命名空间包和相对导入 | 理解 petvision.data.datasets 对应哪几个目录与文件 |
| [Python 包配置指南（官方英文）](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) | build-system、project；setuptools 的源码位置配置见下方逐项解释 | 建立项目根目录的 pyproject.toml |
| [本地可编辑安装（pip 官方英文）](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) | Editable installs 小节 | 理解 install -e 与修改源码后的效果 |
| [json（官方中文）](https://docs.python.org/zh-cn/3.12/library/json.html) | dump/load、字典与列表、UTF-8 文件读写 | 保存并重新读取划分索引 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读数据接口，建立最小包配置 |
| 2h | 实现加载、固定划分和 DataLoader |
| 1h | 检查索引、形状、类别与可视化 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 按下面第一步先理解 Python 包，再创建 `src/petvision/`、两个空的 `__init__.py` 和根目录的 `pyproject.toml`。将项目安装到当前 petvision 环境，使脚本能统一导入自己写的数据函数；检查导入路径指向本仓库。

2. 在 `src/petvision/data/datasets.py` 中实现 `build_mnist_loaders()`。先用 `torchvision.datasets.MNIST` 下载并读取官方训练集和测试集，根目录设为 `datasets/raw/`。

3. 从官方训练集的 60,000 张图片中，用种子 42 固定选出 55,000 张用于训练、5,000 张用于验证；保存 `experiments/mnist_mlp/week2/mnist_split.json`，后续读取同一份索引。官方测试集的 10,000 张图片单独保留。

4. 为三份数据分别创建 DataLoader 并返回。训练、验证和测试使用相同固定数值预处理；训练 shuffle=True，验证和测试 shuffle=False，不丢弃尾批。

5. 创建 `scripts/inspect_mnist.py` 调用数据函数。检查索引无重叠、范围正确且覆盖官方训练集；打印一个批次的图片 `[B,1,28,28]`、标签 `[B]`、dtype 和取值范围，显示少量图片及标签。今天只核对测试集来源和长度，不用它选配置。

### 分步操作与自查

<a id="package-setup"></a>
#### 第一步：理解 Python 包，建立目录并验证本地安装

这一步是在整理后续要复用的 Python 代码。第一周的多个脚本放在 scripts 中，可以导入同目录的 gradient_basics；接下来数据、模型、训练函数会被多个脚本和检查共同使用，统一放进 petvision 包，用明确的完整名称导入。

先认识三个概念：一个 `.py` 文件叫模块；包含 `__init__.py` 的目录在本项目中作为普通 Python 包；包中还可以继续放子包。`src` 是存放源码的目录，本项目从 petvision 开始导入。

**1. 按目录图创建文件，两个 __init__.py 暂时留空。**

```text
PetVisionLab/
├── pyproject.toml                 # 本步创建：项目安装配置
├── scripts/
│   └── inspect_mnist.py           # 第五步编写：调用数据函数并检查
└── src/
    └── petvision/                 # 你自己的 Python 包
        ├── __init__.py            # 本步创建：空文件即可
        └── data/                  # 数据子包
            ├── __init__.py        # 本步创建：空文件即可
            └── datasets.py        # 第二至四步编写：加载、划分和批次
```

`__init__.py` 是包的初始化文件，本题先留空，让导入只完成包加载。以后在 datasets.py 写好 build_mnist_loaders 后，脚本可以这样导入：

```python
from petvision.data.datasets import build_mnist_loaders
```

这条路径逐段对应 `petvision/ → data/ → datasets.py → 函数名`。只写 `import petvision` 会加载顶层包；需要某个数据函数时再按完整路径导入。今天两个空文件分别属于 petvision 和 data；后续新增 models、training 等包目录时，也分别添加自己的 __init__.py。

**2. 在项目根目录创建 pyproject.toml，告诉安装工具怎样找到这些代码。**

pyproject.toml 是文本配置文件，使用 TOML 格式：`[名称]` 表示一组配置，下面的 `键 = 值` 描述这组设置。setuptools 是负责查找包、准备安装内容的工具，pip 按这里指定的方式调用它。先使用以下完整最小配置：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "petvision-lab"
version = "0.1.0"
requires-python = ">=3.12"

[tool.setuptools.packages.find]
where = ["src"]
```

逐项理解其作用：

| 配置 | 本题的含义 |
| --- | --- |
| `[build-system]` | 说明准备项目安装时使用什么工具 |
| `requires = ["setuptools>=68"]` | 构建工具需要至少 68 版的 setuptools；这里描述的是构建依赖 |
| `build-backend = "setuptools.build_meta"` | pip 通过 setuptools 提供的这个接口准备项目安装 |
| `[project]` 中的 name | 安装管理时使用的项目名，本题为 petvision-lab |
| version | 给自己的项目标记版本，起始用 0.1.0 |
| requires-python | 声明项目支持 Python 3.12 及以上；当前解释器仍由 Conda 环境提供 |
| `[tool.setuptools.packages.find]` 中的 where | 从 src 目录里查找 Python 包，找到其中的 petvision |

安装名 petvision-lab 来自配置中的 name；导入名 petvision 来自源码包目录，所以代码写 `import petvision`。Conda 环境也叫 petvision，它负责管理 Python 与依赖；这里同名的 Python 包负责组织你写的代码。

**3. 激活环境，在含 pyproject.toml 的项目根目录做本地可编辑安装。**

```bash
conda activate petvision
python -m pip install -e . --no-deps
python -c "import petvision; print(petvision.__file__)"
python -m pip check
```

把安装命令拆开读：

- `python -m pip`：使用当前 Python 解释器对应的 pip，安装到已激活的环境。
- `install`：安装这个本地项目。
- `-e`：editable，即可编辑安装，让 Python 使用开发目录里的源码；修改 src 下的普通 Python 代码后，下次运行会读取修改后的内容。
- `.`：当前目录，也就是含 pyproject.toml 的 PetVisionLab 根目录。
- `--no-deps`：跳过安装项目声明的运行依赖，本项目依赖已在前一天准备；构建工具仍可能按 build-system 准备。

**4. 看导入路径，确认环境找到的是刚建立的包。**

你的本地输出预期类似：

```text
/home/sun/PetVisionLab/src/petvision/__init__.py
```

`petvision.__file__` 是当前加载包的文件路径。看到它指向本仓库 src/petvision，说明后面的 scripts 与 tests 可以统一导入这里的代码。若报 No module named petvision，先核对 sys.executable 属于 petvision 环境、安装命令执行目录正确、src/petvision/__init__.py 存在；若 pyproject.toml 的名称、版本或包配置发生变化，重新执行安装命令。

本步完成时应能解释「目录负责组织代码，配置负责描述项目，安装让当前解释器找到项目包」。数据读取函数在下一步实现。

<a id="mnist-data"></a>
#### 第二步：下载并读取 MNIST，确认数据来源

在 `src/petvision/data/datasets.py` 中实现以下接口；第二至四步的片段是同一个函数的组成部分，导入语句放在文件顶层，下载和数据处理放在函数内：

```python
def build_mnist_loaders(data_dir, split_path, batch_size, seed):
    # 返回 train_loader、val_loader、test_loader
    ...
```

四个参数分别是数据根目录、划分索引 JSON 路径、每批样本数和随机种子。今天由检查脚本传入 `datasets/raw/`、本周的 `mnist_split.json`、128 和 42。

先在文件顶层导入本函数会用到的工具；`json` 和 `Path` 用于下一步保存 / 读取索引：

```python
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
```

在函数内建立固定预处理，再读取两份官方数据：

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])
train_base = MNIST(
    root=data_dir, train=True, download=True, transform=transform,
)
test_set = MNIST(
    root=data_dir, train=False, download=True, transform=transform,
)
```

- `root=data_dir`：本题传入 `datasets/raw/`，torchvision 在其下管理 `MNIST/` 数据目录。
- `download=True`：缺少数据时下载；已有完整数据时直接使用。
- `train=True`：读取官方 60,000 张训练图片；下一步再从中划出训练 / 验证子集。
- `train=False`：读取官方 10,000 张测试图片；它不表示验证集。
- `ToTensor()`：把本题的灰度图片转为 `[1,28,28]` 浮点 Tensor，并将像素从 0–255 缩放到 0–1。
- `Normalize(...)`：逐像素计算 `(x-0.5)/0.5`，得到约 [-1,1] 的范围。三份数据使用同一固定变换。

先在函数中临时打印 `len(train_base)`、`len(test_set)`，应为 `60000`、`10000`。再取 `image, label = train_base[0]`：图片应为 `[1,28,28]` 的 float32 Tensor，label 是 0–9 中的一个 Python 整数。单张标签在 DataLoader 组成批次后才变为 `[B]` 的 long Tensor。

<a id="mnist-split"></a>
#### 第三步：从 MNIST 官方训练集生成并保存划分索引

本步只划分第二步的 train_base：60000 张图片中的 55000 张用于训练，5000 张用于验证；官方 test_set 的 10000 张图片保持独立。保存的是图片编号，不是复制图片文件。

下面三段代码按顺序放在 `build_mnist_loaders()` 内，接在创建 train_base / test_set 之后、第四步创建 DataLoader 之前。文件顶层的 json、Path、torch 和 Subset 已在第二步导入。如果你已经写了 `split_path = Path(split_path)`，这一行保留一次即可。

**1. 判断已有文件：有则读取，没有才生成。**

```python
split_path = Path(split_path)
n_total = len(train_base)
if n_total != 60000:
    raise ValueError(f"MNIST 官方训练集应有 60000 条，实际为 {n_total}")

# 首次生成划分；已有文件时读取，避免覆盖之前的实验条件。
split_exists = split_path.exists()
if split_exists:
    with split_path.open("r", encoding="utf-8") as file:
        split_data = json.load(file)
else:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    split_data = {
        "dataset": "mnist",
        "version": 1,
        "seed": seed,
        "n_total": n_total,
        "train_indices": indices[:55000],
        "val_indices": indices[55000:],
    }
```

- Path 将字符串路径转成路径对象；exists() 判断划分文件是否已经存在。
- randperm 生成 0..59999 的不重复随机排列；前 55000 个编号和其余 5000 个编号来自同一个排列，因此天然不重叠。
- Generator 只控制本次划分的随机性；本题 seed=42。后续训练批次的随机顺序由第四步的另一条 Generator 控制。
- tolist() 将 Tensor 转为普通整数列表，便于写入 JSON。
- 已有文件时不重新随机划分，读取保存的编号；仅设置同一 seed 不能替代保存和复用索引。

JSON 中的 dataset、version、seed、n_total 说明这份划分适用于哪个数据集和设置；train_indices / val_indices 保存两份实际编号。

**2. 对新生成和已读取的划分，执行同一套检查。**

```python
# 读取和首次生成的划分都使用相同的完整性检查。
if not isinstance(split_data, dict):
    raise ValueError("划分 JSON 的顶层必须是字典")
expected_metadata = {
    "dataset": "mnist", "version": 1, "seed": seed, "n_total": n_total,
}
for name, expected in expected_metadata.items():
    if split_data.get(name) != expected:
        raise ValueError(
            f"划分文件的 {name} 与当前设置不符："
            f"期望 {expected!r}，实际 {split_data.get(name)!r}"
        )

for name, expected_size in (("train_indices", 55000), ("val_indices", 5000)):
    values = split_data.get(name)
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须是列表")
    if len(values) != expected_size:
        raise ValueError(f"{name} 应有 {expected_size} 项，实际为 {len(values)}")
    if not all(type(index) is int for index in values):
        raise ValueError(f"{name} 中的索引必须是整数")
    if not all(0 <= index < n_total for index in values):
        raise ValueError(f"{name} 中存在超出 0..{n_total - 1} 的索引")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} 中存在重复索引")

train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]
train_index_set = set(train_indices)
val_index_set = set(val_indices)
if train_index_set & val_index_set:
    raise ValueError("训练集与验证集的索引存在重叠")
if train_index_set | val_index_set != set(range(n_total)):
    raise ValueError("训练集与验证集的索引未完整覆盖官方训练集")
```

这些检查依次回答：文件格式与当前设置是否匹配、数量是否为 55000/5000、编号是否为合法整数、各组内部是否重复、两组是否重叠、合起来是否覆盖全部 60000 张训练图片。`&` 是集合交集，`|` 是集合并集。

检查失败时抛出 ValueError，并保留已有文件。比如用相同路径传入不同 seed，不应自动覆盖旧划分；确实要做另一组实验时使用新的 split_path。

**3. 检查通过后保存新文件，再创建两个 Subset。**

```python
# 只有首次生成且检查通过后，才保存到约定路径。
if not split_exists:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as file:
        json.dump(split_data, file, ensure_ascii=False, indent=2)

train_set = Subset(train_base, train_indices)
val_set = Subset(train_base, val_indices)
```

mkdir 创建 JSON 所在目录，json.dump 将已检查的字典写入 UTF-8 文件；只有本次首次生成时才写盘。Subset 把原始 dataset 和编号关联起来：`train_set[i]` 读取的实际图片是 `train_base[train_indices[i]]`，没有复制整份图片。

**本步自查：**

1. 在函数末尾临时打印 `len(train_set), len(val_set), len(test_set)`，应为 `55000 5000 10000`。
2. 首次调用应生成 `experiments/mnist_mlp/week2/mnist_split.json`；再次调用应读回同一组编号，文件内容不变。
3. 使用同一路径但不同 seed，应得到清楚的报错；恢复原 seed 后仍能读取原文件。

完成第三步时函数尚未返回 DataLoader；下一步接上三个 loader 的构造和 return。当前只验证索引与子集，不需要运行模型训练。

<a id="mnist-loaders"></a>
#### 第四步：创建并返回三个 DataLoader

Dataset 提供一张图片与标签；DataLoader 负责按照 batch_size 取出多张并组成批次。沿用第二步文件顶层的导入，在同一个函数内接上：

```python
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    generator=torch.Generator().manual_seed(seed),
    num_workers=0, drop_last=False,
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    num_workers=0, drop_last=False,
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False,
    num_workers=0, drop_last=False,
)
```

上述语句放在 `build_mnist_loaders()` 内，最后写 `return train_loader, val_loader, test_loader`，把三份批次加载器交给调用者。训练 DataLoader 的 Generator 独立于划分索引用的 Generator，分别控制取样顺序和数据划分。

`shuffle=True` 会在遍历训练数据时打乱顺序，但不会改变哪些图片属于训练集。验证和测试保持固定顺序，`drop_last=False` 保留最后不足一个完整批次的数据。

`batch_size=128` 表示每批最多 128 张，`num_workers=0` 表示在当前 Python 进程加载数据。训练 Generator 的状态会随使用推进，所以不同 epoch 通常有不同的取样顺序；从头运行同样的程序时可以复现这一串顺序。验证和测试使用顺序取样，无需通过 Generator 控制随机打乱，所以这里省略它。Generator 也可用于数据加载工作进程的随机种子；后续若启用多进程或随机变换，再一并设置。

#### 第五步：调用函数并检查一批 MNIST 图片

创建 `scripts/inspect_mnist.py`。下面是完整脚本：调用数据函数，打印并检查数量、形状、类型和范围，再保存和显示训练批次中的 8 张图片。

```python
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from petvision.data.datasets import build_mnist_loaders


def main():
    train_loader, val_loader, test_loader = build_mnist_loaders(
        data_dir="datasets/raw",
        split_path="experiments/mnist_mlp/week2/mnist_split.json",
        batch_size=128,
        seed=42,
    )
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    assert len(train_loader.dataset) == 55000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000

    images, labels = next(iter(train_loader))
    print("images:", images.shape, images.dtype)
    print("image range:", images.min().item(), images.max().item())
    print("labels:", labels.shape, labels.dtype)
    print("label range:", labels.min().item(), labels.max().item())
    assert images.shape == (128, 1, 28, 28)
    assert labels.shape == (128,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all().item()
    assert images.min().item() >= -1 and images.max().item() <= 1
    assert labels.min().item() >= 0 and labels.max().item() <= 9

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for index, ax in enumerate(axes.flat):
        image = images[index].cpu() * 0.5 + 0.5
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(labels[index].item()))
        ax.axis("off")
    fig.tight_layout()
    figure_path = Path("experiments/mnist_mlp/week2/figures/mnist_batch.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    print("figure saved:", figure_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- `next(iter(train_loader))` 取一个训练批次。images 为 `[128,1,28,28]`、float32；labels 为 `[128]`、long，标签值在 0..9。函数内部已检查划分索引无重叠、无越界并完整覆盖官方训练集。
- `assert` 表示必须满足的条件；条件不成立会立即报错，避免只看打印结果而漏掉错误。这些断言针对今天固定的 batch_size=128。
- `fig` 是整张画布，`axes` 是 2×4 个子图组成的数组；`axes.flat` 逐个取出子图。`enumerate` 同时给出编号 0..7，用来读取对应图片和标签。
- `image * 0.5 + 0.5` 撤销前面的归一化；`squeeze(0)` 将 `[1,28,28]` 变成 `[28,28]`。`imshow` 用灰度显示，`vmin=0, vmax=1` 固定黑白对应的数值范围；标题是正确标签，今天还没有模型预测。
- `tight_layout()` 调整子图间距；`savefig()` 保存图片；`show()` 在支持图形窗口的环境中显示；`close(fig)` 在显示结束后关闭画布。
- 文件末尾的入口保护负责调用 `main()`；其他文件导入此脚本时，不会自动下载或画图。今天只核对测试集来源和长度，不遍历测试集计算模型指标。

打印的数据集数量应为 `55000 5000 10000`。batch_size 是每批样本数，和数据集总数不同；`drop_last=False` 时，训练最后一批为 88 条，验证最后一批为 8 条，测试最后一批为 16 条。

脚本先将图片保存到 `experiments/mnist_mlp/week2/figures/mnist_batch.png`，再显示窗口。没有图形界面的终端可用 `MPLBACKEND=Agg python scripts/inspect_mnist.py` 运行；此时 show 可能提示不能显示窗口，保存的图片仍可在编辑器中打开。

#### 第六步：保存可复用条件

将今天使用的条件记录在 `experiments/mnist_mlp/week2/README.md`：data_dir、split_path、seed、batch_size、num_workers、drop_last，以及 ToTensor → Normalize(mean=(0.5,), std=(0.5,)) 和各 loader 的 shuffle 设置。保留划分 JSON 和图片。Day 4 再将这些字段接入训练 YAML，Day 5 保存正式运行的配置快照。

### 运行与验收

```bash
python scripts/inspect_mnist.py
```

首次运行下载数据并保存 `mnist_split.json`；第二次读取同一份索引。检查打印的 train/val/test 数量分别为 `55000/5000/10000`、训练批次通常为 `[128,1,28,28]`，并查看显示的 8 张图与标签。下载失败先核对原始报错和数据路径，避免把缺失文件误当模型错误。

- [ ] 训练/验证索引可复用且互不重叠，官方测试集没有混入训练。
- [ ] 知道 batch size 与样本数的区别；能解释 num_workers=0 的起始选择。
- [ ] 脚本可从根目录运行，包导入不依赖手工修改 sys.path。

<a id="day-3"></a>
## Day 3：MLP、logits 与交叉熵

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [搭建模型（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html) | nn.Module、Flatten、Linear、ReLU | 实现 MLP.forward |
| [优化模型参数（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) | CrossEntropyLoss 与训练步骤 | 理解 logits 到 loss |
| [pytest 入门（官方英文）](https://docs.pytest.org/en/stable/getting-started.html) | 第一个测试、assert、pytest.raises | 将形状和数值检查写成 test_ 函数 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读模型与交叉熵，手算一个三分类例子 |
| 2h | 实现 MLP 与一次反向传播 |
| 1h | 检查 logits、标签、梯度和参数量 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `src/petvision/models/cv/mlp.py`，实现输入展平、两层隐藏层与输出层，按本周固定结构返回 `[B,10]` logits。

2. 用三个手写 logits 和一个正确类别，理解 `-log softmax(logits)[target]`。交叉熵直接接收 logits；准确率用 argmax，展示概率时再 softmax。

3. 创建 `tests/test_cv_training.py` 的首批检查：batch=1 与 batch=4 均得到 `[B,10]`；CrossEntropyLoss 为有限标量；反向后参与计算的参数梯度存在且有限。

4. 打印可训练参数量，解释每个 Linear 的 weight 与 bias 形状。将模型移到选定 device，图片与标签也一起迁移。

### 分步操作与自查

#### 第一步：按尺寸写出 MLP

在项目根目录下新建 `src/petvision/models/` 和其中的 `cv/` 目录，各放一个空的 `__init__.py`，再创建 `src/petvision/models/cv/mlp.py`。已有目录和文件沿用，保留其中的实现。目录关系为：

```text
src/petvision/models/
├── __init__.py
└── cv/
    ├── __init__.py
    └── mlp.py
```

在 mlp.py 顶层写 `from torch import nn`，定义继承 `nn.Module` 的 `MLP` 类。`__init__` 先调用 `super().__init__()`，再创建 Flatten、Linear、ReLU 等层；`forward(self, images)` 按以下顺序调用这些层并返回 logits。模型文件只保留导入和类定义，实例化、损失计算与反向检查放到第三步的测试文件中。

```text
图像            [B,1,28,28]
Flatten         [B,784]
Linear + ReLU   [B,256]
Linear + ReLU   [B,128]
Linear          [B,10]
```

构造函数声明层，forward 只做输入到 logits 的计算。优化器、数据加载和 loss 留在训练入口。今天可以先把上述固定尺寸写在类中，用 `MLP()` 创建模型。Day 4 [第四步](#mnist-configurable-interfaces)先把固定尺寸改成构造参数，默认值分别为 784、(256,128)、10，验证改造有效后再写 YAML；之后创建和重建模型都使用这些配置参数。此结构参数量应为 235146，第四步再检查。

#### 第二步：手算一次交叉熵

这段是理解损失的独立练习；先在 Python 终端中 `import torch` 后运行，打印 loss.item() 核对结果。将正确类的 logit 从 0 改成 2，再计算一次，确认损失下降。本步使用手写的三个 logits；下一步才检查实际 MLP 的十类输出。

logits=`[0,0,0]` 时三类概率均为 1/3，标签为第 0 类时 loss=`-log(1/3)≈1.098612`。提高正确类 logit 后，loss 应下降。

```python
logits = torch.tensor([[0.0, 0.0, 0.0]])
target = torch.tensor([0], dtype=torch.long)
loss = torch.nn.functional.cross_entropy(logits, target)
```

target 保存类别编号；训练不必先做 one-hot。预测使用 argmax(dim=1)，展示概率时再用 softmax；交叉熵接收原始 logits。

#### 第三步：完成最小前反向检查

本步只回答一个问题：**第一步写的 MLP，能否把图片变成分类输出、计算损失，并为模型参数算出梯度？** 先在 CPU 上使用随机图片和随机标签检查计算过程。

**1. 创建测试文件，明确代码写在哪里。**

在项目根目录创建 `tests/`，在其中创建 `test_cv_training.py`。下面是一份完整的最小示例，写入这个测试文件；模型类仍保存在 `src/petvision/models/cv/mlp.py`，通过 import 使用。

```python
import torch

from petvision.models.cv.mlp import MLP


def test_mlp_forward_backward():
    for batch_size in (1, 4):
        # 每次新建模型，分别检查单张图片和一个小批次。
        model = MLP().cpu()
        images = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,), dtype=torch.long)

        # 前向：每张图片得到 10 个类别的 logits。
        model.zero_grad()
        logits = model(images)
        assert logits.shape == (batch_size, 10)

        # 损失：将 logits 与正确类别编号比较，得到一个有限标量。
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        # 反向：为参与计算的模型参数计算梯度。
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"

        print(f"batch={batch_size}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")
```

**2. 按计算顺序理解这段检查。**

| 顺序 | 代码 | 检查什么 |
| --- | --- | --- |
| 准备输入 | 创建 images 和 labels | B 张单通道 28×28 图片，每张对应一个 0–9 的标签 |
| 前向 | `logits = model(images)` | 输出形状是 `[B,10]`；B=1 时也保留 batch 维 |
| 计算损失 | `cross_entropy(logits, labels)` | loss 是一个标量，且不是 NaN 或无穷大 |
| 反向 | `loss.backward()` | 根据本次损失计算模型参数的梯度 |
| 检查梯度 | 遍历 `model.named_parameters()` | 本题 MLP 的所有可训练参数都有梯度，且每个梯度元素有限 |

`assert 条件` 表示“这个条件必须成立”，不成立时测试报错。`torch.isfinite` 检查数值是否既不是 NaN，也不是正负无穷；`.all()` 要求所有元素都符合。本题允许某些梯度元素为 0，ReLU 会使部分路径的梯度为 0。

每个 batch_size 都重新创建模型；这里的 zero_grad 用于明确“清梯度 → 前向 → loss → backward”的顺序。反向只计算梯度，更新参数需要 Day 4 学习的优化器与 step。

**3. 从项目根目录运行测试。**

```bash
python -m pytest tests/test_cv_training.py -q -s
```

pytest 会自动执行名称以 `test_` 开头的函数，不需要在文件末尾手动调用，也不需要加 main 入口。`-q` 精简测试报告，`-s` 显示 print 输出；直接用 `python tests/test_cv_training.py` 不会自动执行这些测试函数。

只写入上述一个测试函数时，应看到 batch=1、batch=4 两行输出，以及 `1 passed`。两个 batch 的 logits 形状分别为 `(1,10)`、`(4,10)`；随机输入的 loss 没有固定数值，也不要求本次随机分类准确。

#### 第四步：检查参数量和设备

第三步通过后，把下面两个测试函数**追加到同一个 `tests/test_cv_training.py`**，保留前面的导入和测试函数。

**1. 核对参数量。**

```python
def test_mlp_parameter_count():
    model = MLP()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", parameter_count)
    assert parameter_count == 235146
    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape))
```

Linear(in,out) 的 weight 形状为 `[out,in]`，bias 为 `[out]`，参数数为 `in*out+out`。三层分别为 `784*256+256`、`256*128+128`、`128*10+10`，合计 235146；Flatten 与 ReLU 没有可训练参数。

**2. 核对模型和数据能一起迁移设备。**

```python
def test_mlp_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    images = torch.randn(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,), dtype=torch.long).to(device)

    logits = model(images)
    assert logits.shape == (4, 10)
    assert logits.device == images.device == labels.device
    assert all(parameter.device == images.device for parameter in model.parameters())
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss).item()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"
    print("device:", logits.device)
```

这里选择 CUDA（可用时）或 CPU；模型、图片和标签要在同一设备。图片保持浮点类型，标签保持 long，迁移设备不会把标签变成浮点数。第三步始终检查 CPU 路径，本测试额外检查当前可用的设备。

### 运行与验收

从项目根目录再次执行：

```bash
python -m pytest tests/test_cv_training.py -q -s
```

如果文件中只有本节提供的三个测试函数，应显示 `3 passed`。参数量为 `235146`；第二步手写三分类例子的 loss 约为 `1.098612`，它与第三步随机图片的十分类 loss 是两项不同的检查。

若形状检查失败，先检查 Flatten 是否保留 batch 维；若矩阵乘法报错，检查第一层是否接收 784 个特征；若标签类型报错，检查其形状为 `[B]`、dtype 为 long；若某参数没有梯度，按报错中的参数名检查它是否参与 forward。

- [ ] B=1 和 B=4 均输出 `[B,10]`，loss 是有限标量。
- [ ] backward 后，所有可训练参数的梯度存在且有限。
- [ ] 参数量为 235146，能解释 Linear 的 weight / bias 形状。
- [ ] 模型、图片和标签处于同一设备，标签保持 long。
- [ ] 能区分前向计算、损失计算、反向计算梯度和优化器更新参数。

<a id="day-4"></a>
## Day 4：训练循环与小批过拟合

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [优化循环（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/optimization_tutorial.html) | zero_grad、backward、step、评估循环 | 实现训练与验证函数 |
| [保存加载（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | state_dict 和验证模式 | 准备 checkpoint 接口 |
| [itertools.islice（Python 官方中文）](https://docs.python.org/zh-cn/3.12/library/itertools.html#itertools.islice) | `from itertools import islice`、`islice(loader, 2)` | smoke 模式只遍历前两个批次 |
| [Python argparse（官方中文）](https://docs.python.org/zh-cn/3.12/library/argparse.html)、[PyYAML 文档](https://pyyaml.org/wiki/PyYAMLDocumentation) | 复习 Day 1 的参数解析；新增 required、store_true、互斥参数组和 type=int；PyYAML 看 safe_load 和 safe_dump | 将参数与配置字典接入训练入口 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 复习一轮训练与评估的顺序 |
| 2h | 按最小示例实现参数/配置读取、训练与验证 |
| 1h | 进行 smoke 与 64 样本过拟合检查 |
| 0.5h | 固定正式预算，准备日志与曲线输出 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 按第一步创建 `src/petvision/utils/seed.py`、`src/petvision/training/cv_trainer.py`，实现随机种子、train_one_epoch 与 evaluate。训练切 train、验证切 eval/no_grad，图片和标签迁移到同一设备。

2. 按第二步在 `tests/test_cv_training.py` 追加统计检查：loss 按每批实际样本数加权，accuracy 按总正确数/总样本数计算；尾批和部分遍历的 n_samples 均应正确。

3. 按第三步创建 `scripts/check_mnist_training.py`，先独立运行两个训练批次和两个验证批次，再新建模型，固定同一批 64 条训练样本、关闭随机增强并循环约 200 次更新。目标训练准确率接近 100%；后续再把这两项接入训练入口的 --smoke / --overfit-samples 分支。

4. 按第四步先改造 `src/petvision/models/cv/mlp.py` 和 `src/petvision/data/datasets.py`，使模型尺寸、验证集大小、归一化与 num_workers 由参数控制；保留旧调用的默认值，验证参数生效后再创建 `configs/cv/mlp_mnist.yaml`。第五步创建 `scripts/train_cv.py` 并解析配置，第六步将字段传给这些接口。

5. 过拟合关卡通过后固定 5 epochs 正式预算，准备每轮 train/val loss、accuracy 的日志与画图函数。Day 5 接通 checkpoint 保存后，再从新模型和优化器开始正式训练并生成 `history.csv`、曲线，避免先训练完才发现没有保存 best。

### 分步操作与自查

<a id="training-seed"></a>
#### 第一步：准备随机种子工具，再写训练 / 评估函数

创建 `src/petvision/utils/`、`src/petvision/training/` 及各自空的 `__init__.py`，再创建 `src/petvision/utils/seed.py`，实现以下小函数；导入它不会立即改变随机状态，由训练入口在创建新模型前调用：

```python
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

`random`、NumPy 全局 RNG 和 Torch 分别有自己的随机状态；独立的 NumPy `default_rng` 或 Torch Generator 仍需在创建时单独传种子。固定 seed 用于在相同条件下重复实验，跨硬件或版本仍需核对数值差异，参见 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

接着创建 `src/petvision/training/cv_trainer.py`，编写：

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    # 返回按样本汇总的 loss、accuracy
    ...

def evaluate(model, loader, criterion, device):
    # eval 模式、关闭梯度、相同统计口径
    ...
```

两个函数统一返回字典 `{"loss": 平均损失, "accuracy": 总正确数/总样本数, "n_samples": 实际样本数}`，数值转为 Python float / int，方便后续写文件。

本步 criterion 使用 `torch.nn.CrossEntropyLoss(reduction="mean")`。每批用 `loss.item()` 取出损失数值，再乘该批实际样本数累计；循环结束后除以实际处理的样本总数。`n_samples` 在每批按标签数量累加，不直接取整个 dataset 的长度，这样后续 smoke 只遍历两批时也能正确统计。返回字段统一写 `n_samples`。

每批顺序为清梯度→前向→loss→backward→step。model、images、labels 在同一设备；验证不调用 optimizer。

训练函数每次进入时显式 `model.train()`；evaluate 每次进入时 `model.eval()` 并在 `torch.no_grad()` 中前向。验证结束后，下一轮训练函数会恢复 train 模式。用一次真实训练批检查至少一个参与参数的更新前后数值不同；用一次验证检查参数及已有梯度没有变化，不能只以 loss 有数值作为循环正确的证据。

#### 第二步：检查整轮指标是否统计正确

这一步检查第一步的函数是否正确汇总多个批次。先做纸上计算，再用一份答案已知的小数据实际调用 evaluate；模型训练能否学会内容，留到第三步检查。

**1. 先算清楚为什么不能直接平均每批指标。**

假设两批分别有 128、16 条样本，平均 loss 分别为 1、3，正确数分别为 64、16：

```text
总样本数 = 128 + 16 = 144
平均 loss = (128*1 + 16*3) / 144 ≈ 1.222222
accuracy = (64 + 16) / 144 ≈ 0.555556
```

不能把 loss 算成 `(1+3)/2=2`，也不能把 accuracy 算成 `(0.5+1)/2=0.75`，因为两批的样本数不同。你在第一步写的累计逻辑应当与上面的公式一致。

**2. 在现有测试文件中追加一个真正调用 evaluate 的测试。**

打开 `tests/test_cv_training.py`，保留 Day 3 的三个函数和顶层的 `import torch`，在文件末尾追加下面这个函数。这里使用 3 条小数据、batch_size=2，故意产生大小为 2 和 1 的两个批次，方便核对尾批；它与上面的 144 条手算例子是同一种统计规则。

```python
def test_evaluate_metrics():
    from itertools import islice

    from torch.utils.data import DataLoader, TensorDataset
    from petvision.training.cv_trainer import evaluate

    # 为了单独检查统计，直接准备 3 条已知的分类分数。
    logits = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
    model = torch.nn.Identity()
    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))
    expected_loss = criterion(logits, labels).item()
    assert metrics["n_samples"] == 3
    assert abs(metrics["loss"] - expected_loss) < 1e-6
    assert abs(metrics["accuracy"] - 1 / 3) < 1e-6

    # 只读取第一批时，应统计 2 条，而不是整个 dataset 的 3 条。
    partial = evaluate(model, islice(loader, 1), criterion, torch.device("cpu"))
    assert partial["n_samples"] == 2
    assert abs(partial["loss"] - criterion(logits[:2], labels[:2]).item()) < 1e-6
    assert partial["accuracy"] == 0.5
    print("full:", metrics)
    print("first batch:", partial)
```

这里没有使用 MNIST 或 MLP：TensorDataset 将每条分数与标签配成一对，Identity 原样返回输入分数，让我们专门检查统计。三条预测均为类别 0（相同最高分时 argmax 取第一个位置），只有第一条正确，所以整体 accuracy=1/3；第一批两条中有一条正确，所以部分遍历的 accuracy=1/2。

`expected_loss` 将三条数据一次性交给交叉熵，作为全体样本平均损失的参考答案；evaluate 则分两批累计。如果误把每批 loss 直接平均，两个结果就会不同。再用 islice 只读第一批，检查 n_samples 是否真的只计已处理的 2 条。

**3. 从项目根目录运行。**

```bash
python -m pytest tests/test_cv_training.py -q -s
```

若文件中只有 Day 3 的三个测试和新增这一个，应显示 `4 passed`。本例 full 的 loss 约为 1.171074、accuracy 约为 0.333333、n_samples=3；first batch 的 loss 约为 0.693147、accuracy=0.5、n_samples=2。失败时先检查 loss 是否乘了该批样本数、分母是否为实际样本数，以及返回字段是否写成 n_samples。

#### 第三步：先跑两批，再让模型学会固定 64 条样本

这一步分成两项：smoke 检查数据、模型、优化器与训练 / 验证函数能否接通；固定小样本拟合检查参数更新是否真正有效。今天先按顺序运行这两项，后面第四至第六步再将它们接入 YAML 和 `train_cv.py` 的参数分支。

**1. 创建独立检查脚本。**

创建 `scripts/check_mnist_training.py`，写入下面的完整示例。它直接使用已经实现的 MLP() 和四参数数据函数，不依赖尚未创建的训练 YAML；后续扩展构造函数时保留默认值，这份检查仍能使用。

```python
from itertools import islice
import math

import torch
from torch.utils.data import DataLoader, Subset

from petvision.data.datasets import build_mnist_loaders
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import train_one_epoch, evaluate
from petvision.utils.seed import set_seed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    train_loader, val_loader, _ = build_mnist_loaders(
        data_dir="datasets/raw",
        split_path="experiments/mnist_mlp/week2/mnist_split.json",
        batch_size=128,
        seed=42,
    )
    criterion = torch.nn.CrossEntropyLoss()
    print("device:", device)

    # A. smoke：只训练两批、验证两批，确认接口能接通。
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_metrics = train_one_epoch(
        model, islice(train_loader, 2), optimizer, criterion, device
    )
    val_metrics = evaluate(model, islice(val_loader, 2), criterion, device)
    print("smoke train:", train_metrics)
    print("smoke val:", val_metrics)
    assert train_metrics["n_samples"] == 256
    assert val_metrics["n_samples"] == 256
    assert math.isfinite(train_metrics["loss"])
    assert math.isfinite(val_metrics["loss"])

    # B. 过拟合：新建模型和优化器，反复学习同样的 64 条训练样本。
    set_seed(42)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    fixed_set = Subset(train_loader.dataset, list(range(64)))
    fixed_loader = DataLoader(fixed_set, batch_size=64, shuffle=False)
    images, labels = next(iter(fixed_loader))
    fixed_batch = [(images, labels)]

    initial = evaluate(model, fixed_batch, criterion, device)
    print("overfit initial:", initial)
    for step in range(1, 201):
        train_one_epoch(model, fixed_batch, optimizer, criterion, device)
        if step % 20 == 0:
            metrics = evaluate(model, fixed_batch, criterion, device)
            print(
                f"step={step}, loss={metrics['loss']:.6f}, "
                f"accuracy={metrics['accuracy']:.2%}"
            )

    final = evaluate(model, fixed_batch, criterion, device)
    assert final["n_samples"] == 64
    assert final["loss"] < initial["loss"]
    assert final["accuracy"] >= 0.98, "固定小样本尚未拟合好，请检查训练流程"


if __name__ == "__main__":
    main()
```

**2. 先理解 smoke 部分，再运行整个脚本。**

```bash
python scripts/check_mnist_training.py
```

islice(train_loader,2) 只提供前两个训练批次，验证同理。batch_size=128，因此两份 n_samples 都应为 256；loss 必须有限，准确率只需是合法比例，不设达标目标。smoke 的目的在于确认代码能执行，不代表模型已经训练好。

**3. 看懂固定 64 条样本是怎么做到的。**

- `Subset(train_loader.dataset, list(range(64)))` 选取训练子集中的前 64 条；这些编号属于训练子集，不会取验证或测试图片。
- 只执行一次 `next(iter(fixed_loader))`，把这 64 张图片和标签读到 images / labels。MNIST 当前只有固定的数值预处理，没有随机增强。
- `fixed_batch = [(images, labels)]` 是只包含一个批次的列表，可以反复遍历；不要把它换成每轮重新抽样。
- 200 次调用 train_one_epoch 时，每次只处理这一个批次，因此总计 200 次参数更新。每 20 次用 evaluate 在同样的 64 条训练样本上检查拟合结果。
- 过拟合检查重新设置 seed、新建模型和优化器，避免继承前面 smoke 的训练状态。

**4. 判断是否通过。**

loss 应从初始值明显下降，固定 64 条训练样本的 accuracy 应接近 100%；示例用至少 98% 作自查门槛。2026-10-25 在 RTX 4060 Laptop GPU 上验证本例，200 次更新后 loss 约为 0.000319，accuracy=100%；具体数值可能随环境变化，不要求逐位相同。这里证明模型能记住这 64 条训练数据，不能据此推断验证或测试准确率。

若未通过，依次检查标签是否正确、每次是否仍用同一批样本、梯度是否清零、是否执行 backward / optimizer.step，以及学习率是否为 0.001。需要继续定位时，打印梯度范数和更新前后参数差。

本脚本只打印检查结果，不保存正式 history、配置快照、曲线或 checkpoint。后续将两项逻辑分别接入 `scripts/train_cv.py` 的 `--smoke` 和 `--overfit-samples 64` 分支；检查模式结束后返回，正式训练按 Day 5 安排，重新创建模型和优化器。

<a id="mnist-configurable-interfaces"></a>
#### 第四步：先改造模型与数据接口，再写 MNIST 配置

YAML 只保存配置值。要让它控制程序，必须先有接收参数的模型 / 数据函数，再由训练入口读取 YAML 并传入参数。本步按“改造接口 → 验证参数生效 → 写配置”的顺序完成，第六步才创建完整训练对象。

**1. 将固定 MLP 改成接收尺寸参数的模型。**

打开 `src/petvision/models/cv/mlp.py`，把原来的 `__init__(self)` 和写死尺寸的层替换为下面的实现。forward 仍然只返回 logits。hidden_dims 接收非空的尺寸列表或元组，通过循环创建隐藏层；本周基线配置仍使用 [256,128]，即两个隐藏层。

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(256, 128), num_classes=10):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("MLP 需要隐藏层尺寸")
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        for index in range(len(hidden_dims) - 1):
            self.net.add_module(
                str(len(self.net)),
                nn.Linear(hidden_dims[index], hidden_dims[index + 1]),
            )
            self.net.add_module(str(len(self.net)), nn.ReLU())
        self.net.add_module(
            str(len(self.net)), nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, images):
        logits = self.net(images)
        return logits
```

默认的 MLP() 仍创建 784→256→128→10，因此 Day 3 的检查和独立检查脚本可以继续使用。

循环要写 `range(len(hidden_dims)-1)`，因为整数本身不能被遍历。`add_module(name,module)` 接收字符串名称和实际的层对象；这里用 `str(len(self.net))` 为新层编号，保持默认模型的 state_dict 键名与原实现一致。最后一层先创建 `nn.Linear(hidden_dims[-1],num_classes)` 再加入 Sequential，输出层后面不加 ReLU。hidden_dims=(8,) 和 (8,4,2) 分别创建一个、三个隐藏层；空序列明确报错。传入 hidden_dims=(512,256) 后，才会创建不同宽度的隐藏层；只在 YAML 写这个值而不传给构造函数，不会改变模型。

**2. 将 MNIST 数据函数中的固定设置改成参数。**

打开 `src/petvision/data/datasets.py`，把接口扩展为下面的形式；省略号位置保留原函数体，并逐项按后面的清单修改，不要把函数体替换成省略号：

```python
def build_mnist_loaders(
    data_dir, split_path, batch_size, seed,
    validation_size=5000, num_workers=0, mean=(0.5,), std=(0.5,),
):
    ...
```

- 将 Normalize 中写死的 (0.5,) 改为传入的 mean、std：`transforms.Normalize(mean=mean, std=std)`。
- 在已有 n_total 后检查 `0 < validation_size < n_total`，不满足则抛出 ValueError；计算 `n_train = n_total - validation_size`。
- 首次划分时，train_indices 改为 `indices[:n_train]`，val_indices 改为 `indices[n_train:]`。
- 索引长度检查中的 55000/5000 分别改为 n_train / validation_size，保留整数、范围、唯一性、互不重叠与完整覆盖检查。
- 三个 DataLoader 的 num_workers 都读取函数参数；训练 shuffle、独立 Generator、drop_last 和返回值保持原有约定。
- 已有 JSON 时仍然读回并核对，包括 val_indices 的长度与 validation_size 一致；不覆盖旧文件。本步不必改变 JSON 格式，原来的 55k/5k 文件应能按默认值继续读取。

这样，原有四参数调用仍使用 5000、0、(0.5,)、(0.5,)；后续训练入口也能明确传入 YAML 中的数值。官方测试集仍是独立的 10000 条，不参与验证集大小的调整。

**3. 验证“传了参数，结果真的改变”。**

完成模型改造后，在 `tests/test_cv_training.py` 追加下面的测试，保留所有已有测试：

```python
def test_mlp_configurable_dimensions():
    model = MLP(input_dim=4, hidden_dims=(8, 4), num_classes=3)
    images = torch.randn(2, 1, 2, 2)
    logits = model(images)
    assert logits.shape == (2, 3)
    assert sum(p.numel() for p in model.parameters()) == 91
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all().item()
```

这是独立的小形状检查，不是更改 MNIST 正式配置；4→8→4→3 的参数数为 `(4*8+8)+(8*4+4)+(4*3+3)=91`。它与已有的默认模型 235146 参数检查一起，确认新参数被使用、旧默认行为仍兼容。

数据接口也做小检查：默认参数读取原划分时仍为 55000/5000/10000；使用独立临时 split_path 和 validation_size=10000 时应为 50000/10000/10000；相同旧路径传入不同 validation_size 应明确报错，而不是覆盖原划分。把 mean/std 改为 (0.0,)/(1.0,) 后，图片应保留 ToTensor 的 0–1 范围，验证传入值确实影响预处理。检查使用临时路径，不修改正式基线的 YAML 或划分文件。

```bash
python -m pytest tests/test_cv_training.py -q -s
```

接口检查通过后再继续。若出现 unexpected keyword argument，说明被调用类或函数的签名尚未完成改造；不能通过删掉入口参数而让配置悄悄失效。

<a id="mnist-config"></a>
**4. 将已经支持的参数写入 YAML。**

创建 `configs/cv/mlp_mnist.yaml`，以下作为本周起始配置。YAML 的缩进表示字典层级；后续周沿用 `data.batch_size`、`data.split_path`、`training.learning_rate` 等字段名，不另起 `batch`、`lr` 的同义字段。

```yaml
model:
  name: mlp
  input_dim: 784
  hidden_dims: [256, 128]
  num_classes: 10
data:
  name: mnist
  root: datasets/raw
  split_path: experiments/mnist_mlp/week2/mnist_split.json
  validation_size: 5000
  batch_size: 128
  num_workers: 0
  normalize:
    mean: [0.5]
    std: [0.5]
training:
  seed: 42
  device: auto
  dtype: float32
  epochs: 5
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
experiment_dir: experiments/mnist_mlp/week2
output_dir: outputs/mnist_mlp
```

`data.root` 是下载位置，`data.split_path` 复用 Day 2 的索引。数据函数中先 ToTensor，再按 normalize 配置做 Normalize；`validation_size` 仅用于首次划分，已有 JSON 则读取并核对。`device: auto` 由入口解释为“CUDA 可用则用 CUDA，否则 CPU”，它不是 PyTorch 自带的设备名；本周 `dtype: float32` 指模型及图片，分类标签仍为 long。

配置与代码的对应关系如下；第五步只读取成 cfg，第六步才将这些值传入并创建对象：

| YAML 字段 | 实际使用位置 |
| --- | --- |
| model.input_dim / hidden_dims / num_classes | MLP 的三个构造参数 |
| data.validation_size / num_workers | build_mnist_loaders 的划分与 DataLoader 参数 |
| data.normalize.mean / std | 数据函数内部的 Normalize |
| model.name / data.name / training.optimizer | 入口选择 MLP、MNIST、Adam；本周遇到其他名字明确报错 |

读取后的 `cfg["data"]["batch_size"]` 应为整数 128，`cfg["training"]["learning_rate"]` 为数值 0.001。按配置选择 Adam，weight_decay=0；模型尺寸、归一化和路径都从相应字段读取，避免 YAML 写了一套、代码实际用了另一套。

<a id="cli-config"></a>
#### 第五步：把命令行参数与配置字典接起来

创建 `scripts/train_cv.py`，将下面的最小入口示例写入这个文件；先理解并运行它，再把普通训练函数接到 main 中。它目前只解析和打印，不会自动训练模型。

沿用 [Day 1 的参数与路径用法](#environment-output)：`--config` 同样用 `type=Path` 接收文件路径，`required=True` 表示必须提供；互斥参数组限制两个检查模式不能同时选择。`action="store_true"` 把是否出现某个开关转为布尔值，`type=int` 将样本数转换为整数。

```python
import argparse
from pathlib import Path
import yaml


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--overfit-samples", type=int)
    args = parser.parse_args(argv)
    if args.overfit_samples is not None and args.overfit_samples <= 0:
        parser.error("--overfit-samples 必须大于 0")
    return args


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件顶层必须是字典")
    print(cfg["model"]["name"], cfg["data"]["batch_size"])
    print(args.smoke, args.overfit_samples)
    # 下一步：按模式创建数据、模型和优化器，调用训练/评估函数


if __name__ == "__main__":
    main()
```

`--config` 后面的值是文件路径；`--smoke` 是开关，不需要再写 True；`--overfit-samples 64` 在 Python 中对应 `args.overfit_samples == 64`。两个检查模式互斥，都不传时表示正式训练。所有相对路径统一从项目根目录解释，按下方命令核对参数和两个检查模式。

<a id="training-objects"></a>
#### 第六步：把配置连接到数据、模型和优化器

本步以[第四步接口改造与检查](#mnist-configurable-interfaces)已经完成为前提：MLP 已接收三个尺寸参数，build_mnist_loaders 已接收 validation_size、num_workers、mean、std，默认调用检查仍然通过。现在将以下步骤接到 scripts/train_cv.py 的 main 读出 cfg 之后，逐字段创建对象；不要把本步的参数调用提前接到仍使用旧签名的类或函数上。本周仍只使用 FP32。

导入语句放在 `train_cv.py` 顶层：

```python
import torch
from petvision.data.datasets import build_mnist_loaders
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import train_one_epoch, evaluate
from petvision.utils.seed import set_seed
```

在 main 中按下面顺序创建对象：

```python
set_seed(cfg["training"]["seed"])
device_name = cfg["training"]["device"]
if device_name == "auto":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
train_loader, val_loader, test_loader = build_mnist_loaders(
    data_dir=cfg["data"]["root"],
    split_path=cfg["data"]["split_path"],
    batch_size=cfg["data"]["batch_size"],
    seed=cfg["training"]["seed"],
    validation_size=cfg["data"]["validation_size"],
    num_workers=cfg["data"]["num_workers"],
    mean=cfg["data"]["normalize"]["mean"],
    std=cfg["data"]["normalize"]["std"],
)
model_cfg = cfg["model"]
model = MLP(
    input_dim=model_cfg["input_dim"],
    hidden_dims=model_cfg["hidden_dims"],
    num_classes=model_cfg["num_classes"],
).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=cfg["training"]["learning_rate"],
    weight_decay=cfg["training"]["weight_decay"],
)
```

`model.name` 用于入口选择模型，不是传给 MLP 的构造参数。在读取配置后、创建数据和模型前，明确检查本周仅支持 model.name=mlp、data.name=mnist、training.optimizer=adam、training.dtype=float32；其他值抛出 ValueError，不能继续按旧实现运行。第三周再扩展 CNN / ResNet 和 AdamW 分支。

对象创建后先判断检查模式：smoke 将传给两个循环的批次各限制为最多 2 个，顶层 `from itertools import islice` 后可用 `islice(loader, 2)`；overfit 固定 64 条训练样本反复更新。 进入 overfit 分支时先检查 args.overfit_samples 不超过 len(train_loader.dataset)，超出时明确报错，再创建 Subset；参数大于 0 的检查由第五步的 parse_args 完成。正式分支才逐 epoch 调用 `train_one_epoch(model, train_loader, optimizer, criterion, device)` 和 `evaluate(model, val_loader, criterion, device)`。今天先完成检查模式，正式分支的落盘与训练在 Day 5 接通。

<a id="run-modes"></a>
#### 第七步：隔离检查模式与正式结果

本步继续修改 `scripts/train_cv.py`，在已有训练循环外补上结果管理。顺序是：检查模式结束后返回 → 正式分支保存配置 → 每轮收集并打印指标 → 训练结束后统一保存 CSV 和曲线。以下代码先接好，正式五轮训练仍等 Day 5 加入 checkpoint 保存后再执行。

| 模式 | 实际执行 | 输出规则 |
| --- | --- | --- |
| smoke | 新建模型/优化器，最多 2 个训练批次和 2 个验证批次 | 只打印本次检查结果 |
| overfit | 新建模型/优化器，固定 64 条训练样本、约 200 次更新 | 只打印本次检查结果 |
| 正式训练 | 新运行从固定种子、新模型和新优化器开始，按完整配置训练 | 写配置快照、history 和曲线；Day 5 再保存 best/last |

**1. 给两个检查分支加 return。**

保留现有的 if / elif / else 结构，在下面两个位置各补一行 `return`：

- smoke 分支：放在 `print("smoke val:", val_metrics)` 的下一行，与这条 print 同缩进。
- overfit 分支：放在 `print("overfit final:", final)` 的下一行，与这条 print 同缩进。它位于 200 次更新的 for 循环之外。

当前 if / elif / else 已将三个分支互斥隔离，即使省略显式 return，检查模式也不会进入 else。这里加 return 是为了在检查结束时明确退出 main，避免以后追加在整段条件语句之后的保存操作被执行。创建 experiment_dir、写 config.yaml 和 history 的操作均放在正式 else 内；smoke / overfit 不创建这些结果，已有正式文件也保持不变。

**2. 在文件顶层补两个导入。**

```python
import csv
import matplotlib.pyplot as plt
```

原来的 `from pathlib import Path`、yaml、torch 等导入继续保留。

**3. 在 parse_args 后、main 前定义保存与绘图函数。**

下面两个函数放在文件顶层，不缩进到 main 或 parse_args 里面：

```python
def save_history(history, experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy",
    ]
    with (experiment_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, experiment_dir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set(title="Loss", ylabel="Cross entropy")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val")
    axes[1].set(title="Accuracy", ylabel="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()

    figure_dir = Path(experiment_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "training_curves.png", dpi=150)
    plt.close(fig)
```

save_history 接收本次运行完整的 history 列表，用固定五列写 CSV。当前小实验在训练循环结束后只调用一次，以 w 模式写入“表头 + 全部记录”；如果文件已经存在，会覆盖旧内容。若训练在循环中途异常退出，本次尚未保存的指标只在内存中，不会写入 CSV。

未来耗时较长的实验可以改为逐轮追加：每轮只传入最新的 row，用 a 模式和 writer.writerow(row) 写入，表头仅在新文件中写一次。新实验与恢复训练还需要区分输出文件，避免把不同运行的指标混在一起。当前步骤先保持结束后统一保存。

plot_history 只读取 history 中已经算好的指标：左图为 train/val loss，右图为 train/val accuracy，横轴都是 epoch。准确率保存为 0–1 的比例；画图不会重新训练或重新评估。savefig 后关闭画布，不需要 show。

**4. 替换现有 else 分支里的内容。**

保留 `else:` 这一行，将它原来的内容（包括已写的 config.yaml 保存代码）整体替换为下面片段。下方代码放在 main 的 else 内，整体缩进 8 个空格；不要重复保留旧的训练循环。

```python
epochs = cfg["training"]["epochs"]
if type(epochs) is not int or epochs <= 0:
    raise ValueError("training.epochs 必须是正整数")

experiment_dir = Path(cfg["experiment_dir"])
experiment_dir.mkdir(parents=True, exist_ok=True)
with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as file:
    yaml.safe_dump(cfg, file, allow_unicode=True, sort_keys=False)

history = []
for epoch in range(1, epochs + 1):
    train_metrics = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )
    val_metrics = evaluate(model, val_loader, criterion, device)
    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }
    history.append(row)
    print(
        f"epoch={epoch}, "
        f"train_loss={row['train_loss']:.6f}, "
        f"train_accuracy={row['train_accuracy']:.2%}, "
        f"val_loss={row['val_loss']:.6f}, "
        f"val_accuracy={row['val_accuracy']:.2%}"
    )
    # Day 5 在每轮验证后接入 checkpoint 保存，再进行正式五轮训练。

save_history(history, experiment_dir)
plot_history(history, experiment_dir)
```

这段代码按以下顺序执行：

1. 检查 epochs 为正整数，再创建本次配置指定的 experiment_dir。
2. 用 yaml.safe_dump 保存本次 cfg，便于之后核对实际训练条件。
3. history 初始为空列表；每轮完成训练和验证后，整理同一轮的五个字段，追加一条 row。
4. 每轮打印同一组实际指标，便于观察训练过程。
5. 全部 epoch 结束后，用 save_history 一次保存全部记录，再使用这份 history 画图；两个函数都放在 for 循环外。

`range(1, epochs+1)` 让记录中的 epoch 从 1 开始；配置 epochs=5 时，CSV 最终应有一行表头和五行数据。默认输出位置为：

```text
experiments/mnist_mlp/week2/
├── config.yaml
├── history.csv
└── figures/
    └── training_curves.png
```

当前只实现指标与图表保存，模型权重保存按 Day 5 接入；正式运行时，两者必须来自同一次训练。smoke 和 overfit 各自启动一次新进程时，现有 main 会先固定种子、创建新模型和优化器，因此检查状态不会继承到后续正式运行。

**5. 今天先验证检查模式，不启动正式五轮。**

完成上述修改后，仍使用下方“运行与验收”里的 --smoke、--overfit-samples 64 和 pytest 命令。检查运行前后，config.yaml、history.csv、曲线和 checkpoint 不应因检查而新增或被改写。

这组示例已用临时数据验证配置快照、两轮 CSV、曲线生成和画布关闭，并检查了有 / 无已有正式结果时两种检查模式的文件隔离。临时验证未运行正式五轮，也未生成正式模型权重。Day 5 接入 checkpoint 保存后，再用不带检查开关的命令完成正式运行。

### 运行与验收

```bash
python scripts/train_cv.py --help
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --smoke
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml --overfit-samples 64
python -m pytest tests/test_cv_training.py -q
```

`--help` 应列出本日实现的参数。smoke 最多处理两个训练和两个验证批次；过拟合检查反复使用固定 64 条样本。两者只打印检查结果，正式五轮训练在明天接通保存后执行。若出现 NaN，先查输入与 loss 是否有限，再检查学习率和梯度；若 loss 不降，确认清梯度、反向与参数更新都确实发生。

- [ ] smoke 跑通，过拟合检查证明模型能记住固定小样本。
- [ ] 训练与验证模式、设备迁移和按样本加权统计正确。
- [ ] 模型和数据接口已参数化，默认调用仍兼容，改变参数的小检查通过。
- [ ] YAML 字段已传给实际对象，入口按配置创建模型 / 数据 / 优化器；smoke/overfit 不覆盖正式结果。
- [ ] 日志与曲线输出函数已接好，正式训练安排在 Day 5 保存流程完成后。

<a id="day-5"></a>
## Day 5：最佳 checkpoint 与加载一致性

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [保存与加载模型（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | 保存 state_dict、重建模型、eval 与断点信息 | 比较加载前后的验证 logits |
| [pytest 临时目录（官方英文）](https://docs.pytest.org/en/stable/how-to/tmp_path.html) | tmp_path 参数和路径拼接 | 保存加载检查写入临时目录 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读 checkpoint 与最佳模型选择 |
| 2h | 实现保存、加载和独立 evaluate 入口 |
| 1h | 完成正式训练并检查加载一致性 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 在 `src/petvision/utils/checkpoint.py` 保存模型 state_dict、模型配置、epoch、验证指标与种子。验证准确率改善时写 `outputs/mnist_mlp/best.pt`；每轮写 last.pt，避免只保留内存中会继续改变的最佳权重引用。

2. 创建 `scripts/evaluate.py`：按配置重建模型，接受 `--checkpoint` 与 `--split`；先仅评估 val。评估不得更新模型，打印指标及实际样本数。

3. 固定一个验证批次，在 eval 模式下比较保存前与重新加载后的 logits 和验证指标，FP32 起始误差参考 `1e-6`，记录设备差异造成的合理数值波动。

4. 接通保存后，从新模型和优化器完成正式 5 epochs；补充 checkpoint 与验证统计检查，汇总该次运行的训练曲线、最佳 epoch 和一例实际调试问题。

### 分步操作与自查

#### 第一步：定义 best 和 last

best 按验证 accuracy 选择，持平时可约定保留更早一次；last 是最近完成的一轮。规则写入 README，不预设 best 一定在最后。

今天会修改三个位置：新建 `src/petvision/utils/checkpoint.py`，修改已有 `scripts/train_cv.py` 的正式 else 分支，新建 `scripts/evaluate.py`；测试继续添加到 `tests/test_cv_training.py`。本周 checkpoint 用于重建模型和评估，第七周再补优化器与随机状态以支持等价续训。

Day4 未配套权重的五轮结果现已归档到 `experiments/mnist_mlp/week2/archive/day4_without_checkpoint/`，其中保存 config.yaml、history.csv 和 training_curves.png；mnist_split.json 仍在本周主目录。下面的新运行会重新生成主目录日志，并保存与其对应的模型权重。后续若重跑，先归档需要保留的配置、指标、曲线和权重。

保存到磁盘时写 state_dict；若先在内存保留最佳状态，需要深拷贝，避免后续训练改变同一份引用。

<a id="checkpoint-io"></a>
#### 第二步：保存到文件，再重建并加载模型

创建 `src/petvision/utils/checkpoint.py`，先定义两个文件读写函数。它们只负责读写传入的字典，模型选择和 best 判断仍由入口负责，后续视觉 / 语言任务可以复用：

```python
from pathlib import Path
import torch


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)
```

在 train_cv.py 顶层导入 `from petvision.utils.checkpoint import save_checkpoint` 和 hashlib。正式 else 内、for 循环前设置 `best_accuracy = float("-inf")`，并计算 `split_sha256 = hashlib.sha256(Path(cfg["data"]["split_path"]).read_bytes()).hexdigest()`。数据函数已经读取或建立该文件；hash 用来识别实际索引，避免同名文件内容变化后误用。

每轮 evaluate 返回后，在 Day 4 留出的 checkpoint 注释处建立下方字典；此处仍位于 for 内，cfg、model、epoch、val_metrics 均来自当前 main：

```python
checkpoint = {
    'model_state': model.state_dict(),
    'model_config': cfg['model'],
    'config': cfg,
    'epoch': epoch,
    'val_accuracy': val_metrics['accuracy'],
    'val_loss': val_metrics['loss'],
    'seed': cfg['training']['seed'],
    'split_sha256': split_sha256,
}
```

紧接着调用刚定义的保存函数。best_accuracy 必须在循环外初始化，否则每轮都会被重置；history 和曲线仍按 Day 4 的约定在循环结束后统一保存：

```python
output_dir = Path(cfg["output_dir"])
save_checkpoint(checkpoint, output_dir / "last.pt")
if val_metrics["accuracy"] > best_accuracy:
    best_accuracy = val_metrics["accuracy"]
    save_checkpoint(checkpoint, output_dir / "best.pt")
```

加载流程：读字典→按配置新建模型→load_state_dict→迁移设备→eval→关闭梯度推理。加载一个字典并不会自动生成模型对象。在 evaluate.py 顶层导入 MLP、torch 和 `from petvision.utils.checkpoint import load_checkpoint`，确定 device 后，重建的关键操作为：

```python
checkpoint = load_checkpoint(args.checkpoint)
model_cfg = checkpoint["model_config"]
model = MLP(
    input_dim=model_cfg["input_dim"],
    hidden_dims=model_cfg["hidden_dims"],
    num_classes=model_cfg["num_classes"],
)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device).eval()
```

创建 `scripts/evaluate.py`，复用 Day 4 的参数解析方式读取 `--config、--checkpoint、--split`；`--config`、`--checkpoint` 为必填路径，`--split` 限定 val/test。文件顶层从 petvision 包导入 MLP、数据函数和 `training.cv_trainer.evaluate`，在 main 中核对 YAML 中模型配置与 checkpoint 一致，再用保存的划分创建 loader，按 split 选择 val_loader 或 test_loader，最后调用已有 evaluate，打印 loss、accuracy、n_samples。文件末尾加入口保护；它不创建优化器，也不调用训练循环。

按下面顺序完成这个新文件，不能只复制上面的加载片段就直接运行：

1. 顶层补齐 argparse、hashlib、Path、yaml、torch，以及上述包导入。parse_args 中 `--config` / `--checkpoint` 使用 `type=Path, required=True`；`--split` 使用 `choices=("val", "test"), default="val"`。
2. main 读取 YAML 并用 load_checkpoint 读权重；检查 `cfg["model"] == checkpoint["model_config"]`，当前 model.name 必须为 mlp。对照保存的 config，核对 data.name、validation_size、normalize 和 training.seed；读取实际 split 文件并比较 split_sha256。发现不匹配明确报错，不重新生成另一份划分。data.root 可按实际缓存位置调整，batch_size 和评估设备可改变。
3. 按 Day 4 解释 training.device，当前 dtype 仅接受 float32。调用 build_mnist_loaders 时逐项传入 root、split_path、batch_size、seed、validation_size、num_workers、mean、std，调用方式与训练入口一致。
4. 用上面的加载片段建立模型，然后按下方代码选择 loader 和打印结果。这里 `evaluate` 继续使用已有按样本加权的统计实现。
5. 文件末尾写 `if __name__ == "__main__": main()`。先运行 `python scripts/evaluate.py --help` 核对参数，再在训练产生权重后进行独立验证。

配置字典在 Python 中的变量名是 cfg，但保存进 checkpoint 时用的键名是 config，读取应写 `checkpoint["config"]`，不能写成 checkpoint["cfg"]。设备选择也要放在解析参数、读取 YAML 之后，先赋值 `device_name = cfg["training"]["device"]`。

下面片段接在 main 的模型加载之后，train_loader 不参与评估：

```python
loader = val_loader if args.split == "val" else test_loader
criterion = torch.nn.CrossEntropyLoss()
metrics = evaluate(model, loader, criterion, device)
assert metrics["n_samples"] == len(loader.dataset)
print({"split": args.split, "epoch": checkpoint["epoch"], **metrics})
```

读取这类由 Tensor、字典和普通数值组成的 checkpoint 时，显式使用 `torch.load(checkpoint_path, map_location="cpu", weights_only=True)`，先在 CPU 上重建并加载，再迁移到评估设备。`map_location="auto"` 不是有效的 Torch 设备映射；YAML 中的 device=auto 是本项目入口自己解释的约定。这里先加载到 CPU，再由 model.to(device) 选择实际设备。`map_location` 解决保存设备与加载设备不同的问题；`model.eval()` 切换层的运行模式，`torch.no_grad()` 关闭本次推理的梯度记录，两者都要处理。

#### 第三步：固定验证批次比较

保存前切 eval 并记录一批 logits；新建模型加载后比较同一批的最大绝对差。差别大时检查结构、归一化、dropout 模式与 checkpoint 路径。

完整验证还应有一致的样本数、loss 和 accuracy。今天使用 `--split val`，最终测试按 Day 6 安排。

保存加载检查按「固定一批输入 → 原模型 eval 得到 logits_before → 保存 → 新建同结构模型并加载 → 同输入得到 logits_after」执行。固定 CPU FP32、关闭随机性时，打印 `(logits_before-logits_after).abs().max().item()` 并检查小于 `1e-6`；不能拿 last 的当前输出与另一个 epoch 的 best 比。

在已有 `tests/test_cv_training.py` 中补充保存加载检查，测试函数可声明 `def test_checkpoint_roundtrip(tmp_path):`，pytest 会提供一个 Path 临时目录，checkpoint 写到 `tmp_path / "model.pt"`。另用人为验证准确率序列 `[0.80,0.85,0.82]` 核对选择逻辑：best 应对应第 2 轮，last 对应第 3 轮。临时检查通过后才运行正式五轮训练。

测试从新 checkpoint 模块导入实际 save_checkpoint / load_checkpoint，保存原模型的真实 state_dict 后新建 MLP 并加载。best / last 检查也读取实际生成的两个文件、核对 epoch；再补一轮 accuracy=0.85，验证持平不会替换较早 best。不要只比较两个相同字典或文件是否存在，那不能证明模型恢复与选择规则正确。

#### 第四步：输出应能回答什么

打印 best epoch、验证指标、保存路径与加载误差。把真实数值填入周总结，说明模型选择依据。

### 运行与验收

```bash
python -m pytest tests/test_cv_training.py -q
python scripts/train_cv.py --config configs/cv/mlp_mnist.yaml
python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split val
```

先通过保存加载的小规模检查，再从新模型正式训练。训练应写出 `config.yaml`、五行 epoch 指标、训练曲线和 `best.pt/last.pt`；独立评估应覆盖 `5000` 条验证样本，并与 best 所在 epoch 的验证结果对应。若文件不存在，先查保存分支和 `output_dir`，不要把尚未保存误判为加载 API 出错。

- [x] best 按验证准确率选择，last 保存最后一轮；本次两者均为第 5 轮，已核对权重一致。
- [x] 独立进程重建模型加载 best，完整验证集 5000 条的 loss / accuracy 与训练日志一致（2026-10-27）。
- [ ] 知道 eval() 与关闭梯度分别解决什么问题。

<a id="day-6"></a>
## Day 6：配置冻结、独立测试与周总结

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [保存加载教程（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | 推理入口与最终评估 | 完成独立测试 |
| [PyTorch 基础总览（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/basics/intro.html) | 回顾本周不熟悉的小节 | 有限参考重写训练步骤 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 冻结配置并进行独立测试 |
| 1.5h | 检查误分类与复现关键结果 |
| 1h | 有限参考重写训练核心与形状解释 |
| 0.5h | 补充每周 README 和环境快照 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 固定本周结构、epoch 与验证选出的 best checkpoint，首次在官方测试集评估，保存 `experiments/mnist_mlp/week2/test_metrics.json` 与少量误分类图。测试后保持该模型配置，后续尝试另列实验。

2. 检查测试样本数为 10,000，指标注明 test；97%+ 仅是参考目标，若偏低先解释流程与错误，不通过反复看测试集来挑参。

3. 关掉参考，在草稿中重写一次清梯度→前向→loss→反向→更新，解释 `[B,1,28,28]→[B,784]→[B,10]`。不额外要求每日记录文件。

4. 更新本周 README：真实版本、划分、运行命令、最佳 epoch、验证/测试结果与一个失败分析；保存实际 Conda/pip 版本清单，确认未把原始数据和权重当小型结果提交。

### 分步操作与自查

#### 第一步：冻结配置再评估测试集

把最终 YAML、best 路径、验证指标和版本一一对应。测试只做前向，使用与验证相同的确定性预处理。

test_metrics.json 至少包含 split、n_samples、loss、accuracy、checkpoint 和配置标识。整体 accuracy 为总正确数/10000，不能只取最后一批。

今天补全 `evaluate.py` 的测试输出：当 `--split test` 时，将这些字段写入配置指定的 `experiment_dir/test_metrics.json`，少量误分类保存到 `figures/mistakes.png`。目录在保存前创建；`--split val` 继续用于独立验证，不覆盖 test 指标。

在 evaluate.py 顶层导入 json。沿用 Day 5 的完整 metrics，在 main 的评估调用之后、`if args.split == "test":` 内构造输出并保存：

```python
result = {
    "split": args.split,
    **metrics,
    "checkpoint": str(args.checkpoint),
    "checkpoint_epoch": checkpoint["epoch"],
    "config": str(args.config),
    "split_sha256": checkpoint["split_sha256"],
}
result_path = Path(cfg["experiment_dir"]) / "test_metrics.json"
result_path.parent.mkdir(parents=True, exist_ok=True)
with result_path.open("w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
```

#### 第二步：查看失败样例

保存少量错误图片的真实类、预测类和分数，观察字迹、类别相似或预处理问题。若低于参考目标，先检查训练验证与加载流程，配置调整回到训练/验证进行。

已有 evaluate 只返回汇总指标，不返回图片。保持它的接口，在 evaluate.py 顶层新增 `collect_mistakes(model, loader, device, limit=8)`：切 eval，在 no_grad 下再次遍历同一个 test_loader；用 `predicted = logits.argmax(dim=1)` 和 `predicted != labels` 找错分位置，每条保存 CPU 图片、真实类、预测类和展示用 softmax 分数。收满 limit 后返回列表；没有错分则返回空列表。完整指标已由前面的 evaluate 统计过 10000 条，这次只采集展示样例，可提前结束，不能用展示批次重新覆盖 metrics。

在 main 的 test 分支调用该函数，再用 `plt.subplots(2,4,figsize=(10,5))` 绘图；先关闭所有坐标轴，再把收集到的样例依次填入，少于 8 张时留空。mean/std 从 cfg["data"]["normalize"] 读取，用 `image.new_tensor(values).view(-1,1,1)` 分别转为与 CPU 图片同类型的通道 Tensor，再计算 `image*std+mean`；不要直接让图片乘 YAML 列表。MNIST 去掉单通道维后 imshow(cmap="gray", vmin=0, vmax=1)；标题写真实类、预测类与分数。保存到 `experiment_dir/figures/mistakes.png` 后关闭画布。若列表为空，在结果中记录错误样例数为 0，不生成误导图片；重跑时检查旧图是否属于本轮。

#### 第三步：有限参考写出两个循环

在草稿中分别写一次训练批次与验证批次，解释 zero_grad、eval 和 no_grad；说明 55k、5k、10k 各用于什么决定。

让草稿实际执行一批：训练后至少一个参与参数发生变化；验证在 eval/no_grad 下参数不变、loss 有限，输出为 `[B,10]`。然后回看原实现核对遗漏。重写的检查使用临时模型，不覆盖正式训练结果。

#### 第四步：补齐复现说明

在周 README 写真实安装命令、显存、划分、配置、best epoch、测试结果和一个调试案例。保留未理解的点，下一周优先补齐；不增加每日记录文件。

### 运行与验收

```bash
python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split test
```

核对终端输出和 `test_metrics.json` 都写明 `split=test`、`n_samples=10000`，检查误分类图中的真实类与预测类。将这轮实际配置和结果写入周总结；测试准确率以 `97%+` 为参考，流程与解释同样要通过验收。

- [ ] 测试集独立使用，结果明确来自哪个 checkpoint 与冻结配置。
- [ ] 能独立解释训练、验证、测试和保存加载的完整流程。
- [ ] 每周 README 可指导重跑，未完成项如实记录并顺延。

## 本周结束检查

- [ ] NumPy 梯度与 autograd 对照通过，小批过拟合检查通过。
- [ ] 固定 MNIST 划分、训练验证曲线、best/last 与加载一致性齐全。
- [ ] 配置冻结后有独立测试结果；能解释 logits、交叉熵与模式切换。
- [ ] 环境、运行命令、结果与实际调试案例集中在本周 README。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第三周](week3.md)。

