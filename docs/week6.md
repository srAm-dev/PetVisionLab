# 第六周逐日学习指南：字符级 Bigram 与 Mini-GPT

> 更新日期：2026-10-25<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-3)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：固定文本划分，完成同一字符 tokenizer 下的 Bigram 与小 GPT、验证曲线、保存加载和自回归生成，解释下一 token 预测。  
> 环境：本地 Conda `petvision`、RTX 4060；起步 2 层、C=128、4 heads、context=128、batch=16，FP32，按实际显存与耗时调整。

## 怎么使用这份指南

先确认[上一周指南](week5.md)的基础关卡已通过。第四周多头 Attention 和 causal Mask 应已通过测试；能独立实现 PyTorch 训练、验证与 checkpoint。视觉结果可保留，本周切换到文本任务。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/mini_gpt/week6/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── src/petvision/
│   ├── data/text.py                         # Day 1–2：原文划分、token 窗口
│   ├── training/lm_trainer.py              # Day 2–4：单步训练与完整验证
│   └── models/llm/
│       ├── tokenizer.py                     # Day 1：字符 tokenizer
│       ├── bigram.py                        # Day 2：Bigram
│       └── mini_gpt.py                      # Day 3：因果语言模型
├── configs/llm/{bigram_char,mini_gpt}.yaml    # Day 2–3：同口径配置
├── scripts/
│   ├── prepare_text.py                      # Day 1：下载来源、固定原文划分
│   ├── train_gpt.py                         # Day 2–4：训练与验证
│   └── generate.py                          # Day 5：加载、自回归生成与输出路径
├── tests/test_language_model.py             # Day 2–3：错位、形状、因果性
├── experiments/mini_gpt/week6/
│   ├── README.md                           # 已提供
│   ├── split_manifest.json                 # Day 1：来源、原文区间与 hash
│   ├── char_tokenizer.json                 # Day 1：训练集字符映射
│   ├── results.csv                         # Day 6：同 tokenizer 验证对照
│   ├── samples.jsonl                       # Day 5：真实生成样例
│   ├── runs/{bigram,gpt}/                   # 配置、history、run_summary 与单模型曲线
│   └── figures/loss_curves.png              # Day 4：train/val loss
├── datasets/processed/tinyshakespeare/      # train/val/test.txt，按原文先划分
└── outputs/{bigram_char,mini_gpt}/           # best/last.pt，不上传
```

## 开始前：激活环境并检查已有模块

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

解释器应位于 Conda 的 `petvision` 环境，`petvision.__file__` 应指向当前仓库的 `src/petvision/`。若出现 `No module named petvision`，先确认第二周已建立 `pyproject.toml` 和包目录，再在根目录执行 `python -m pip install -e . --no-deps`；若缺少 torch 或 CUDA 不可用，按[第二周 Day 1 的安装与排错步骤](week2.md#environment)逐项核对。

复用已经通过检查的 Attention、训练和保存加载方法，新增文本数据与模型。 本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

文本采用作者仓库 Tiny Shakespeare，下载后记录实际来源 URL、文件 SHA256 与解码/换行处理方式。按 Unicode 字符位置顺序切成 80%/10%/10% 的 train/val/test，先保存原文，再分别编码和造窗口；窗口不跨边界。字符词表只从 train 建立，额外保留 `<unk>` ID；验证或 prompt 的未见字符映射到 UNK（未知字符占位 ID），并报告 OOV（未收录字符）的数量，已知字符的 encode/decode 应往返一致。

Bigram 与 GPT 使用相同 tokenizer 文件、数据划分、context=128、batch=16、seed=42、AdamW lr=3e-4、weight_decay=0.01、2000 steps、每 200 步验证。GPT 为 2 层、C=128、4 heads、FFN=512、dropout=0.1；Bigram 为 vocab×vocab 转移 logits 的 Embedding。NLL 指负对数似然，即正确目标 token 的概率取负自然对数。验证在相同原文上按固定顺序统计有效目标 token 的总 NLL/总数，关闭 dropout；按[验证分窗规则](#eval-windows)覆盖 N-1 个目标，尾窗单独评估并记录计数。测试原文留到第七周配置冻结后使用。预算不足时先短运行测时，再同时固定两组实际预算。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | 原文划分、字符词表与数据来源 | 4h | 1h | 固定文本边界与 tokenizer JSON |
| Day 2 | 下一 token 窗口与 Bigram 基线 | 4h | 1h | Bigram 训练与验证入口 |
| Day 3 | 因果 Transformer 与 Mini-GPT | 4h | 1h | 小 GPT、未来隔离与反向检查 |
| Day 4 | 同口径训练、验证与 checkpoint | 4h | 1h | Bigram/GPT 两组曲线与 best |
| Day 5 | 自回归生成与保存加载 | 4h | 1h | 可重跑生成脚本与真实样例 |
| Day 6 | 语言模型比较与核心流程重写 | 4h | 1h | 同 tokenizer 结果表和周总结 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：原文划分、字符词表与数据来源

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | GPT 课程的数据与字符编码部分 | 明确训练文本与字符 ID |
| [Tiny Shakespeare 原文](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | 查看格式、换行和角色名，记录下载来源 | 先划分原文再编码 |
| [Unicode 指南](https://docs.python.org/zh-cn/3.12/howto/unicode.html)、[hashlib](https://docs.python.org/zh-cn/3.12/library/hashlib.html)（官方中文） | UTF-8 编解码、sha256 与 hexdigest；JSON 读写复用第二周 | 区分原始 bytes、实际使用文本及各自 hash |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读字符编码与语言建模任务 |
| 2h | 实现下载记录、原文划分和字符词表 |
| 1h | 检查边界、往返和 OOV |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `scripts/prepare_text.py` 与 `src/petvision/data/text.py`，将原文下载到 `datasets/raw/tinyshakespeare/input.txt`，记录来源、文件 hash 与字符数。固定 80/10/10 原文区间，保存三个 split 文本及 `experiments/mini_gpt/week6/split_manifest.json`，供第七周原样复用。

2. 在 `src/petvision/models/llm/tokenizer.py` 实现 CharTokenizer 的 encode/decode/save/load；只从 train 收集并排序字符，保留独立 `<unk>` ID，保存 JSON。

3. 分别编码 train/val/test，不用验证文本扩充词表；检查训练字符往返一致，统计 val/test 的 OOV 数量。测试集今天仅检查编码可用性，不用其 loss 调参。

4. 用一段短文本展示字符、ID 与解码结果。保留空白换行，不调用 strip() 改变训练内容；下载/规范化规则写入 manifest。

### 分步操作与自查

#### 第一步：区分原文、字符和 token

原文是字符串；字符 tokenizer 把每个已知字符映射为整数 ID；模型接收 ID，并不会直接读取 Python 字符串。空格和换行也是训练内容。

```text
文本： 'ab\na'
映射： {'\n':0, 'a':1, 'b':2}
编码： [1,2,0,1]
解码： 'ab\na'
```

实际映射只来自训练文本，排序后保存，不能每次运行重新按集合遍历顺序编号。另保留 UNK 处理未见字符。

#### 第二步：先划分原文再构造窗口

创建 `src/petvision/data/text.py`，将按字符边界切分的逻辑封装为 `split_text(text)`，返回 train_text、val_text、test_text 三个字符串；明天继续在同一模块加入窗口函数。创建 `scripts/prepare_text.py`，从 `petvision.data.text` 导入 split_text，在 main 中读取原文、调用切分、保存文本和 manifest，末尾添加入口保护。下载与文件写入由脚本执行，导入数据模块不自动下载。

按字符位置切 train/val/test，保存边界与 hash。不能先对整个原文生成大量重叠窗口，再随机把窗口分给 train/val，那样相邻文本可能泄漏。

设完整字符串长度为 L，令 `train_end=int(0.8*L)`、`val_end=int(0.9*L)`，依次保存 `text[:train_end]`、`text[train_end:val_end]`、`text[val_end:]` 到 `datasets/processed/tinyshakespeare/{train,val,test}.txt`。三个区间首尾相接，应覆盖全部字符。

manifest 至少记录 source_url、原始文件 sha256、encoding、换行处理方式，以及三个 split 的路径、`[start,end)` 区间和各自 hash。下载时保留原始 bytes 的 hash；解码、处理换行后再记录实际使用文本的 hash。再次运行优先核对已有文件，避免悄悄覆盖第七周正在使用的数据。

记录下载文件 hash 和读取/换行规范化方式；同一来源 URL 未来内容变化时，hash 能帮助确认是否还是同一份数据。

先用浏览器从原文链接保存到约定路径，或在 prepare_text 中使用标准库下载；两种方式都读取实际落盘 bytes。hash 的最小用法如下：

```python
import hashlib
from pathlib import Path

raw_bytes = Path("datasets/raw/tinyshakespeare/input.txt").read_bytes()
raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
text = raw_bytes.decode("utf-8")
used_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
```

本轮可直接保留原换行，三个 split 用 `Path.write_bytes(part_text.encode("utf-8"))` 写入，父目录先创建。断言三个字符串按顺序拼接等于 text，再计算各 split 的 UTF-8 bytes hash；manifest 的原文件 hash 与实际文本 hash 分别命名。后续读取已有文件时检查 hash 和区间，不能只看文件名相同。

把上述来源、边界和 hash 组织成普通字典，按第二周 json.dump 的 UTF-8 写法保存到 `experiments/mini_gpt/week6/split_manifest.json`，父目录先创建。记录前面计算出的实际字符数和各 split 长度，保存后读回核对三个区间连续、总长为 L。

#### 第三步：写出 tokenizer 的最小接口

创建 `src/petvision/models/llm/` 和空的 `__init__.py`，再创建 `src/petvision/models/llm/tokenizer.py`，将下面的类与 JSON 读写实现写进该文件。在 prepare_text.py 顶层用 `from petvision.models.llm.tokenizer import CharTokenizer` 导入，数据准备脚本只调用它，不再定义另一套字符映射。

```python
class CharTokenizer:
    def __init__(self, chars): ...
    def encode(self, text): ...
    def decode(self, ids): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

构造函数接收已经排序的训练字符列表 chars，建立字符→ID 和 ID→字符的对应关系，同时保存 self.chars、self.unk_id 和 self.vocab_size。prepare_text.py 在完成原文划分后调用 `CharTokenizer(sorted(set(train_text)))`，再保存到 `experiments/mini_gpt/week6/char_tokenizer.json`。

load 是类方法：它从 JSON 读回 chars，用 `cls(chars)` 创建同一种 tokenizer，再核对 version、unk_id，返回这个对象。重新加载时不再读取文本重建词表。已知字符必须往返一致；UNK 映射无法恢复原字符，需要报告 OOV，不把这种输入当成已知字符往返检查。

为避免保存后 ID 改变，本轮可约定 `chars=sorted(set(train_text))`，已知字符 ID 为 `0..len(chars)-1`，UNK ID 为 `len(chars)`，实际 V 为 `len(chars)+1`。JSON 保存 `type="char"、version=1、chars、unk_id`，加载后按列表顺序还原映射，decode 遇到 UNK 可输出 `<unk>` 占位符。用训练文本 `ab\na` 检查已知编码 `[1,2,0,1]`，未见字符 `?` 编为 3，OOV=1；该未知字符的原值不能由 3 恢复。

#### 第四步：检查保存后的同一性

重新加载 JSON，编码同一短文本，ID 序列应一致；保存词表大小与文件 hash，后续 checkpoint 按这个标识加载。

将这些检查接到 `scripts/prepare_text.py` 的保存操作之后，打印已知短文本及往返结果，并分别统计 val/test 中不属于训练字符集合的字符数量；不扩充词表。再次运行时核对已有 split 与 tokenizer，没有条件变化则复用，不静默覆盖跨周使用的文件。

### 运行与验收

```bash
python scripts/prepare_text.py
```

应打印来源、原文长度、三个区间、实际词表大小和 OOV 数，保存三个文本、split manifest 与 `char_tokenizer.json`。重新加载 tokenizer 后，对包含换行的已知字符例子检查往返；val 出现 UNK 时保留数量，不把验证字符加回训练词表。

- [ ] 原文先划分，所有后续模型和 tokenizer 复用相同边界。
- [ ] 词表来源仅为训练文本，OOV 策略明确。
- [ ] tokenizer 可保存加载，已知字符往返不改变空白。

<a id="day-2"></a>
## Day 2：下一 token 窗口与 Bigram 基线

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 字符编码、batch 采样与 Bigram 部分，按主题观看 | 实现输入与目标错位 |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | 对应课程最小 Bigram 实现 | 理解 vocab×vocab logits |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读自回归目标与 Bigram |
| 2h | 实现窗口、Bigram 和交叉熵 |
| 1h | 做形状、错位、验证计数与小批更新检查 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 在 `src/petvision/data/text.py` 构造长度 T+1 的片段，输入为前 T 个 ID，目标为后 T 个 ID；打印一个短例子，逐位置说明预测的是下一个 token。

2. 创建 `src/petvision/models/llm/bigram.py`，使用 Embedding(V,V)，输入 `[B,T]` 输出 `[B,T,V]`。交叉熵将 logits 变为 `[B*T,V]`、target 变为 `[B*T]`，标签 dtype 为 long。

3. 创建 `configs/llm/bigram_char.yaml` 与 `scripts/train_gpt.py`，参数包括 `--config`；在 `src/petvision/training/lm_trainer.py` 实现 train_step 和 evaluate_lm，训练入口负责调用，验证按有效 token 汇总平均 loss。

4. 创建 `tests/test_language_model.py`，检查窗口不跨 split、输入目标错位、batch=1、loss 有限、梯度存在；训练采样使用独立 Generator，验证不推进训练采样状态。今天只用小批数据检查参数能更新，完整 2000 步基线在 Day 4 接通 checkpoint 后运行。

### 分步操作与自查

#### 第一步：用 ABCDE 手算错位窗口

继续修改 `src/petvision/data/text.py`，今天新增 sample_batch 和第四步的 iter_eval_windows。创建 `tests/test_language_model.py`，顶层导入 torch、pytest 和这两个函数；把下面的错位、边界、尾窗计数检查写成测试函数，之后再在同一文件添加模型检查。

设 ID 为 `[0,1,2,3,4]`，T=3，起点 0：

```text
取出的 T+1 个 token：[0,1,2,3]
x：                 [0,1,2]
y：                 [1,2,3]
位置 0 用 token 0 预测 token 1
位置 1 用 token 1 预测 token 2
```

这不是把同一个 token 输入后再让模型复制它。起点必须满足片段完整位于当前 split 内。

长度为 N 的 token 序列中，若每次取 T+1 个 token，起点为 `0..N-T-1`；`torch.randint` 的 high 为 `N-T`，因为上界不包含在结果内。例如 N=5、T=3 时只能从 0 或 1 开始。N 小于 T+1 时明确提示数据不足，不从另一个 split 借 token。

在 data/text.py 实现 `sample_batch(token_ids, context_length, batch_size, generator, device)`。token_ids 为当前 split 的一维 CPU long Tensor；函数用 `torch.randint(0, N-T, (batch_size,), generator=generator)` 产生起点，将每个 `token_ids[start:start+T]` 与 `token_ids[start+1:start+T+1]` 分别用 `torch.stack` 拼为 x / y，再移到 device 后返回这两个 Tensor。stack 是在已有样本外新增一个 batch 维，可查 [torch.stack 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.stack.html)。

调用者在训练开始前创建一次独立 CPU `torch.Generator().manual_seed(42)` 并持续传入，不能每批重新创建或重设种子。输入与 target 都为 long，模型参数是浮点数。本周训练直接从 token 序列取窗口，不需要套用图像 DataLoader。

#### 第二步：理解 Bigram 的能力边界

创建 `src/petvision/models/llm/bigram.py`，顶层写 `from torch import nn`，在其中实现下述 BigramLanguageModel；在 `tests/test_language_model.py` 中从 `petvision.models.llm.bigram` 导入它，补充 B=1、输出形状、loss 和梯度检查。

Embedding(V,V) 可视为一张转移 logits 表，当前 token ID 取出一行，决定下一 token 的分布。它不利用更早的上下文；GPT 将能利用前缀信息。参数量为 V²。在 bigram.py 定义 `BigramLanguageModel(vocab_size)`，构造函数创建 `nn.Embedding(vocab_size, vocab_size)`，forward(token_ids) 返回查表结果；token_ids 必须是 long Tensor。

#### 第三步：把交叉熵维度整理清楚

```text
输入 x / 目标 y      [B,T]，整数 ID
Bigram logits        [B,T,V]
loss 输入 logits     [B*T,V]
loss 输入 target     [B*T]
```

每个位置是一个 V 分类问题。目标 ID 范围必须在 0..V-1，不能先对 logits 做 softmax 再传给 CrossEntropyLoss。

把自回归概率也对应到这个例子：一个 T+1 片段训练的条件概率是各位置下一 token 概率的乘积；取负对数后变为求和，批次 loss 再除以 `B*T`。Bigram 每一项只依赖当前 ID，GPT 每一项可依赖当前可见前缀。用 `[0,1,2,3]` 逐项指出「0→1、1→2、2→3」，解释为什么目标不能与当前位置输入完全相同。

<a id="eval-windows"></a>
#### 第四步：按固定目标区间分窗，完整统计验证 token

训练按随机窗口采样；验证使用确定性的分窗，不抽取随机验证小批替代完整统计。本周不引入 BOS：当前 split 编码后有 N 个 token，索引 0 没有本 split 内的前驱，不计为目标；索引 1..N-1 各预测一次，目标总数应为 N-1。

取 token IDs=`[0,1,2,3,4,5,6,7]`、context=3，按下面的边界取窗口：

| 起点 | 输入 x | 目标 y | 目标数 |
| --- | --- | --- | --- |
| 0 | `[0,1,2]` | `[1,2,3]` | 3 |
| 3 | `[3,4,5]` | `[4,5,6]` | 3 |
| 6 | `[6]` | `[7]` | 1 |

相邻片段共享一个边界 token：上一窗最后的目标成为下一窗第一个输入。目标索引没有重叠，也没有遗漏；尾窗只有一个目标仍要统计。若把 T+1 个 token 整段跳过，窗口之间会漏掉一个预测目标。

可先用以下切片函数核对，再放到 `src/petvision/data/text.py` 中供验证和测试复用。token_ids 是当前 split 的一维 ID 序列：

```python
def iter_eval_windows(token_ids, context_length):
    if context_length < 1:
        raise ValueError("context_length 必须大于 0")
    if len(token_ids) < 2:
        raise ValueError("至少需要两个 token 才能计算下一 token loss")
    for start in range(0, len(token_ids) - 1, context_length):
        chunk = token_ids[start:start + context_length + 1]
        yield chunk[:-1], chunk[1:]
```

真正评估时先将 x、y 作为 long Tensor 移到模型设备，分别补 batch 维为 `[1,t]`；模型返回 `[1,t,V]`。t 可以小于 context，尤其尾窗不能强行 reshape 成固定长度。模型切 eval 并关闭梯度，每窗用 `cross_entropy(..., reduction="sum")` 得到 NLL 总和，累加 `y.numel()` 个目标，最后计算 `total_nll / total_target_tokens`。

上例计数应为 `3+3+1=7=N-1`。若三个窗口的 mean loss 分别为 1、2、4，整体 loss 为 `(3*1+3*2+1*4)/7=13/7`，不能对三个 mean 直接平均。补到现有语言模型检查中：普通尾窗、刚好整除、N=2、不同 context 均满足目标索引恰好覆盖 1..N-1。

该规则会在窗口边界重置模型可见上下文；它是本轮约定的固定评估方式，Bigram/GPT 必须一致，不把它和使用滑动长前缀的评估结果混在一列。每个 split 单独处理，不从 train 借前缀给 val，也不跨入 test。

<a id="text-inputs"></a>
#### 第五步：建立配置，连接已保存的文本与 tokenizer

创建 `scripts/train_gpt.py` 和 `src/petvision/training/lm_trainer.py`。前者沿用第二周 argparse / yaml.safe_load 建立 `--config` 入口与 main 入口保护，负责读文件、创建对象和调度；后者定义下文的 train_step / evaluate_lm，导入时不启动训练。

创建 `configs/llm/bigram_char.yaml`，让训练入口读取本次配置而不是借用分类器的 epoch 参数：

```yaml
model:
  name: bigram
  context_length: 128
data:
  name: tinyshakespeare
  root: datasets/processed/tinyshakespeare
  split_path: experiments/mini_gpt/week6/split_manifest.json
  tokenizer_path: experiments/mini_gpt/week6/char_tokenizer.json
  batch_size: 16
training:
  seed: 42
  device: auto
  dtype: float32
  steps: 2000
  eval_interval: 200
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
experiment_dir: experiments/mini_gpt/week6/runs/bigram
output_dir: outputs/bigram_char
```

读完 YAML 后，先把保存的文本和 tokenizer 接到入口。在 train_gpt.py 顶层导入这些已实现的对象：

```python
from pathlib import Path
import torch
from petvision.models.llm.tokenizer import CharTokenizer
from petvision.models.llm.bigram import BigramLanguageModel
from petvision.data.text import sample_batch
from petvision.training.lm_trainer import train_step, evaluate_lm
from petvision.utils.seed import set_seed
```

main 内读出 cfg 后完成：

```python
tokenizer = CharTokenizer.load(cfg["data"]["tokenizer_path"])
text_root = Path(cfg["data"]["root"])
train_text = (text_root / "train.txt").read_bytes().decode("utf-8")
val_text = (text_root / "val.txt").read_bytes().decode("utf-8")
train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
train_generator = torch.Generator().manual_seed(cfg["training"]["seed"])
```

读取前用 split_manifest 核对文件 hash；按 bytes 解码可保持之前约定的换行。今天测试文本不进入训练 / 验证循环。device 与 set_seed 复用第二周方法，先 set_seed 再新建 `BigramLanguageModel(tokenizer.vocab_size).to(device)` 和 AdamW。一次更新先调用 sample_batch(train_ids, ...)，再将得到的 x / y 交给 train_step；验证把 val_ids 交给 evaluate_lm，按前述 iter_eval_windows 遍历。

在 lm_trainer.py 实现 `train_step(model, x, y, optimizer)`：切 train、清梯度、前向、将 logits / y 展平计算交叉熵、backward、step，返回本批平均 loss 和目标数。`evaluate_lm(model, token_ids, context_length, device)` 切 eval/no_grad，完整累加所有窗口的 NLL 与目标数，返回 loss 和 n_target_tokens，再恢复原模式。两个函数的输入来自不同 split，不能在内部重新随机划分。

两个函数统一返回 `{"loss": float值, "n_target_tokens": int值}`，键名固定；train_step 的数量为 y.numel()，evaluate_lm 为完整分窗的累计目标数。lm_trainer.py 顶层导入 torch、`torch.nn.functional as F` 和 iter_eval_windows；训练 loss 使用 `F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))`，验证使用相同展平方式但 reduction="sum"。evaluate_lm 开始先保存 `was_training = model.training`，在 try 中执行 eval/no_grad，在 finally 中 `model.train(was_training)`，确保退出时恢复原模式。

入口还需校验 training.steps、eval_interval、model.context_length、data.batch_size 为正整数；目前 name 仅支持已实现模型，optimizer 仅支持 adamw，dtype 仅支持 float32。data.name=tinyshakespeare 对应本周保存的原文文件；其他值明确报错。这样修改配置不会被代码静默忽略。

词表大小从 tokenizer.vocab_size 实际读取；`model.context_length` 控制窗口长度，Bigram 本身仍只依赖当前位置。一次 step 指一个训练 batch 的参数更新，2000 steps 不是遍历完整语料 2000 次。今天先实现参数解析与小批更新，正式结果待 Day 4 写出。

本日小批检查要实际做一次更新：先 clone Embedding 的参数，计算有限标量 loss，清梯度、backward、step，确认参数发生变化。训练 Generator 在 evaluate 前后分别 `.get_state().clone()`，用 `torch.equal` 检查状态完全相同；这样才能证明验证没有消耗训练采样流。evaluate 可统一返回 loss 和实际目标 token 数，供训练日志与后续独立检查复用。

### 运行与验收

```bash
python -m pytest tests/test_language_model.py -q
```

今天的检查包含 Bigram 和数据窗口，GPT 明天补充。确认 `[0,1,2,3]` 片段得到错位 x/y，8 个验证 token 的例子完整统计 7 个目标，尾窗参与计算。小批更新使用检查中的临时模型，不写正式 history 和 checkpoint。

- [ ] 能从一个短序列指出每个位置的输入和目标。
- [ ] Bigram logits 与交叉熵维度正确，验证未使用训练窗口。
- [ ] 验证目标索引不重不漏，total_target_tokens=N-1，尾窗参与加权统计。
- [ ] 能解释自回归概率是多个条件概率的乘积，loss 为负对数概率的平均。

<a id="day-3"></a>
## Day 3：因果 Transformer 与 Mini-GPT

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Attention、残差、FFN、位置嵌入与完整模型部分 | 从 Bigram 扩展到 GPT |
| [第四周指南](week4.md) | causal Mask、pre-LN Block 与未来隔离 | 复用已验证的核心 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 画 Decoder-only 张量流 |
| 2h | 实现 GPT 与可学习位置嵌入 |
| 1h | 检查因果性、参数梯度和短训练 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `src/petvision/models/llm/mini_gpt.py` 与 `configs/llm/mini_gpt.yaml`：token embedding + position embedding→两层 pre-LN Block（传入 causal mask）→最终 LayerNorm→Linear(C,V)。

2. 复用第四周 Attention 与 FFN，GPT 无需完整 Encoder–Decoder 或 Cross-Attention；新增位置参数范围为 context=128。

3. 扩展语言模型检查：改变输入后缀，在 eval/dropout=0 下前缀 logits 不变；目标 token 不输入当前位置；输入/输出形状分别为 `[B,T]` 与 `[B,T,V]`。

4. 用极短训练文本做小样本拟合，先检查 loss 能下降和所有活跃参数梯度有限，再按本周固定规模运行。

### 分步操作与自查

#### 第一步：画出自己的小 GPT

创建 `src/petvision/models/llm/mini_gpt.py`，顶层写 `import torch` 和 `from torch import nn`，并从 `petvision.models.transformer.blocks` 导入 EncoderBlock。模型类写在此文件；继续向 `tests/test_language_model.py` 添加本日的形状、因果扰动和梯度检查，从 `petvision.models.llm.mini_gpt` 导入 MiniGPT。

```text
token IDs             [B,T]
token/position embed  [B,T,128]
2 个 causal Block     [B,T,128]
最终 LayerNorm        [B,T,128]
词表输出头            [B,T,V]
```

在 mini_gpt.py 定义 `MiniGPT`，构造参数接收 vocab_size、context_length、embed_dim、depth、num_heads、ffn_hidden_dim 和 dropout。token embedding 用 `nn.Embedding(V,C)`，位置 embedding 用 `nn.Embedding(context_length,C)`；Block 的创建接口见[第四周](week4.md#day-4)。YAML 的 heads 由入口传给 num_heads，name 留在模型选择分支，不能原样把整个 model 字典传给构造函数。

词表 head 的 V 来自 tokenizer，位置嵌入长度来自 context；二者不能混淆。位置 ID 是 0..T-1，输入 T 不得超过配置允许的上下文长度。

#### 第二步：先检查一个位置能看到什么

位置 i 只能使用 x 的 0..i；它的 target 是下一 token。训练一次前向能并行计算所有位置，是因为 mask 保证不会提前读取未来输入。

修改后缀时比较前缀 logits；检查应在 eval 且关闭 dropout 下进行。不要因随机 dropout 让同一输入输出不同而误判为因果泄漏。

取 B=2、T=8、k=3，把一份整数输入 clone 后，仅将 `[:,4:]` 改为 `(原ID+1) % V`。两次使用同一 eval 模型，比较 logits 的 `[:,:4]`，最大绝对差以 `1e-6` 为起始容差；取模保证扰动仍是合法 ID。另检查 T=1 能前向、T 超过 context 时明确报错，避免固定长度演示掩盖位置越界。

#### 第三步：给 forward 和 loss 分工

```python
def forward(self, token_ids):
    # 返回 logits: [B,T,V]
    ...
```

在训练器中根据 logits 和 targets 计算 loss，这样同一模型也能只前向做生成。第四周的 Block 接收布尔 Mask，本轮不新增一个没有实现的 causal=True 参数。在 MiniGPT.forward 中，取 B、T = token_ids.shape，检查长度后创建：

```python
positions = torch.arange(T, device=token_ids.device)
x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
causal_mask = torch.ones((T, T), dtype=torch.bool, device=token_ids.device).triu(1)
for block in self.blocks:
    x = block(x, attn_mask=causal_mask, key_padding_mask=None)
```

上述 self.token_embedding、self.position_embedding 和 self.blocks 均在构造函数中建立；堆叠后接最终 LayerNorm 和词表输出层。每层收到同一个 causal_mask，当前样本不含 padding。ViT 不传 causal mask，GPT 不能沿用它的双向设置。

#### 第四步：先拟合短文本

固定一小段训练文本反复更新，观察 train loss 能否明显下降。若不下降，先查目标错位、mask 方向、optimizer 参数和 step，再扩大文本与训练步数。

可取 train 的前 33 个已编码 ID，构造一组 32 位置的 x/y，使用临时 GPT、dropout=0、固定这组输入约更新 100 次。记录初始与最终 loss，检查最终更低且有限，再检查 token/position embedding、各 Block、输出 head 的梯度存在且有限。这个例子直接放入已有语言模型检查或本地草稿执行，不写正式结果。

从昨天的 Bigram YAML 复制出 `configs/llm/mini_gpt.yaml`，保留完整 data 和 training，将 model 替换为：

```yaml
model:
  name: mini_gpt
  context_length: 128
  embed_dim: 128
  depth: 2
  heads: 4
  ffn_hidden_dim: 512
  dropout: 0.1
```

在 train_gpt.py 顶层导入 MiniGPT，model.name=bigram 时保留昨天的模型分支，name=mini_gpt 时根据本配置创建新的 MiniGPT；两者都调用同一组 train_step / evaluate_lm。

同时将 experiment_dir 改为 `experiments/mini_gpt/week6/runs/gpt`，output_dir 改为 `outputs/mini_gpt`。GPT 复用 Block 的实现并创建新实例，自身管理 token embedding、可学习位置嵌入和最终 LayerNorm，不重复使用教学 Encoder 的正弦位置编码。

### 运行与验收

```bash
python -m pytest tests/test_language_model.py -q
```

加入 GPT 的形状、有限梯度和未来扰动检查：输入 `[B,T]`，输出 `[B,T,V]`；修改位置 k 之后的输入，前 k+1 个位置的 logits 应在容差内一致。小样本拟合只验证学习路径，正式两组训练在明天运行。

- [ ] logits 只依赖当前与过去的输入，未来隔离通过。
- [ ] 能解释 token embedding、position embedding 与输出词表维度。
- [ ] 小模型可以前反向和拟合固定短文本。

<a id="day-4"></a>
## Day 4：同口径训练、验证与 checkpoint

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 保存加载（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | state_dict 与重建模型 | 保存 tokenizer 和模型配置关联 |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | 验证 loss 与训练循环 | 固定比较口径 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 确认两组配置、验证口径和预算 |
| 2h | 完成两组正式训练与保存 |
| 1h | 整理曲线、参数量和耗时 |
| 0.5h | 检查加载后的验证结果 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 两组固定使用同一字符 tokenizer 和验证文本；单独新建模型、优化器与采样 Generator，避免继承另一组状态。

2. 每 200 步验证并写各自 `history.csv`，保留 train loss 的统计窗口和 val loss 的有效 token 数；选择 val loss 最小的 best，另存 last。

3. checkpoint 保存模型 state_dict、config、step、验证指标与 tokenizer 标识/hash；本周先保证加载推理，第七周补齐优化器和随机状态用于续训。

4. 将 tokenizer、配置、参数量、实际训练耗时、峰值显存与两组曲线记录到 week6 实验目录；若 GPT 尚未优于 Bigram，先排查 Mask、错位和训练是否正常，再用机动时间补齐。

### 分步操作与自查

#### 第一步：训练与验证各用独立采样状态

修改已有 `scripts/train_gpt.py`，接通累计 step 循环、定期验证、日志和保存；按需要复用或扩展 `src/petvision/utils/checkpoint.py` 的磁盘读写函数。`src/petvision/training/lm_trainer.py` 继续负责单步更新和完整验证，不能把这些函数另写进模型文件。

训练使用自己的 Generator；验证固定顺序和窗口，不消耗训练采样随机流，否则改变验证频率会改变后续训练批次。

train_step 每次更新前切 train；evaluate_lm 临时切 eval 并关闭梯度，退出时恢复调用前的模式，避免首次验证后 dropout 一直关闭。两组各自从新模型、优化器和 Generator 起步。

#### 第二步：同口径比较两个模型

Bigram/GPT 共享 tokenizer JSON、split manifest、验证窗口规则和 token loss 定义。每组独立创建模型、优化器和采样状态，记录参数量和实际训练预算。

```text
训练日志：step、train_loss、val_loss、val_target_tokens
模型记录：config、tokenizer_hash、split_hash、params
```

训练 loss 可以是最近若干批的平均，验证 loss 是固定验证数据的统计；两者记录时写清窗口，不能直接把单个训练批当全部训练集指标。

本轮可约定 train_loss 为最近 eval_interval 次更新所见目标的平均 NLL，按目标 token 数加权；每个 step 只在 optimizer.step 完成后加 1，随后判断是否验证。2000 步、每 200 步一次时，history 的 step 应为 `200,400,...,2000`，共 10 行数据。last 的 step 与最近保存时实际完成更新次数一致，best 按最小 val_loss 选择。

将下面的调度写在 train_gpt.py 的 main 中。train_ids、val_ids、model、optimizer、device、train_generator 已由 Day 2–3 创建，context_length 和 batch_size 从 cfg 读取；循环前设置区间 NLL 和计数为 0：

```python
interval_nll = 0.0
interval_targets = 0
target_steps = cfg["training"]["steps"]
eval_interval = cfg["training"]["eval_interval"]
for step in range(1, target_steps + 1):
    x, y = sample_batch(train_ids, context_length, batch_size,
                        train_generator, device)
    train_metrics = train_step(model, x, y, optimizer)
    interval_nll += train_metrics["loss"] * train_metrics["n_target_tokens"]
    interval_targets += train_metrics["n_target_tokens"]
    if step % eval_interval == 0 or step == target_steps:
        val_metrics = evaluate_lm(model, val_ids, context_length, device)
        row = {
            "step": step,
            "train_loss": interval_nll / interval_targets,
            "val_loss": val_metrics["loss"],
            "val_target_tokens": val_metrics["n_target_tokens"],
        }
        # 此处追加本条 row，并按第三步判断 best / 保存 last。
        interval_nll = 0.0
        interval_targets = 0
```

末步条件保证 steps 不是 eval_interval 的整数倍时，也验证和保存最后一段。例如 steps=450、eval_interval=200，记录应为 200、400、450 三行，最后一行 train_loss 仅统计最近 50 步。第七周增加 resume 后循环起点再改为 completed_steps+1，并恢复尚未完成区间的累计值。

若统计训练处理量，输入 token 数按实际 batch 累加 `x.numel()`；原文字节数按每个输入 ID 对应字符的 UTF-8 bytes 长度累加，重复采样也重复计数。训练集字符都在训练词表内；下周 BPE 同样累加各 token 的 bytes 长度。静态语料文件大小与训练中累计处理量分别记录。

#### 第三步：按验证 loss 选择 best

验证 loss 下降时保存 best，last 保存最近一次检查点。本轮在每个验证时点和最终步写 last，中断时以文件内 step 为准，最近保存之后尚未落盘的更新需要重跑。本周 checkpoint 至少包含重建所需的结构和词表标识；下周再补完整恢复训练状态。

保存操作复用[第二周 checkpoint 函数](week2.md#checkpoint-io)，在 train_gpt.py 顶层导入 save_checkpoint / load_checkpoint，把判断条件换成 `val_metrics["loss"] < best_val_loss`，循环外将 best_val_loss 初始化为正无穷大，持平保留更早一次。记录 model_state、config、model_config、vocab_size、context_length、tokenizer_path、tokenizer_hash、split_hash、step、val_loss 和 val_target_tokens；hash 来自实际文件，不能只保存路径后省略核对。config.yaml 使用第二周写法，history 则按下方规则逐条追加，不能直接调用使用 epoch / accuracy 字段的视觉 save_history。训练资源记录复用[第三周计时与保存](week3.md#training-resources)，summary 中 epochs 改为 completed_steps，并加入实际累计 input_tokens、input_bytes 和统计范围；保存到各自 experiment_dir/run_summary.json。

checkpoint 同时保存模型配置、实际 V、context_length、tokenizer 路径与 hash、split manifest 的 hash，以及本次 step 和验证 loss。加载时核对这些信息，再按结构重建并读取 state_dict。用新建模型重新计算固定验证集，比较保存前后的 loss 与目标 token 数，不能只检查文件存在。

在 train_gpt.py 顶层新增 `append_lm_history(row, path)`，按[第三周追加函数](week3.md#training-resources)的 a 模式和单条 writerow 实现；列名改为 step、train_loss、val_loss、val_target_tokens，表头只在新文件 / 空文件写一次。正式分支创建各自 experiment_dir / output_dir，保存 config.yaml，在上面 row 创建后立即调用追加函数；每次仅传这一条 row。新运行使用空日志，已有同名结果先归档；第七周的 resume 才继续现有日志。

每个模型训练结束后读取自己 experiment_dir/history.csv，用 step 为横轴、train_loss / val_loss 为纵轴，保存 `experiment_dir/figures/loss_curves.png`。不要复用要求 epoch / accuracy 的视觉 plot_history；在 train_gpt.py 顶层写 `plot_lm_history(history, experiment_dir)`，绘图核心沿用第一周 plot / savefig。

两组都完成后，在草稿或交互环境读取各自 history，再保存合并对照图 `experiments/mini_gpt/week6/figures/loss_curves.png`。单模型训练入口不硬编码另一模型的结果路径，这样先训练 Bigram 时不会因 GPT 日志尚不存在而失败，第七周 BPE 也可直接复用同一入口。

在 `tests/test_language_model.py` 中先用临时目录完成这条加载路径：同模型同批次 eval logits 保存前后最大差小于 `1e-6`；固定验证窗口的目标计数完全一致、平均 loss 差小于 `1e-6`。正式结果则从 best 对应的验证时点取 val_loss 和计数，不拿 last 的输出与 best 混比。

#### 第四步：读曲线并定位异常

train 降而 val 升说明可能过拟合；两者都不降先查实现或优化。GPT 验证未优于 Bigram 时保留真实结果，优先核对泄漏、错位、预算和 checkpoint 是否匹配，不用测试集调整。

### 运行与验收

```bash
python -m pytest tests/test_language_model.py -q
python scripts/train_gpt.py --config configs/llm/bigram_char.yaml
python scripts/train_gpt.py --config configs/llm/mini_gpt.yaml
```

先通过包含保存加载的检查，再分别从新状态运行两组正式实验。每 200 步验证，2000 步预算下应有 10 个验证时点，训练结束时保留两个 best/last 和各自 history。验证频率改变后若下一批训练采样也改变，检查验证是否错误地消耗了训练 Generator。

- [ ] Bigram/GPT 验证 loss 在同一 tokenizer 和目标统计口径下可比较。
- [ ] 模型从新状态训练，best 与 last 不混淆。
- [ ] checkpoint 与 tokenizer 配套，重载后的验证结果一致。

<a id="day-5"></a>
## Day 5：自回归生成与保存加载

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | generate、上下文截断和逐 token 采样 | 实现生成循环 |
| [PyTorch 保存加载（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | 推理 eval 与关闭梯度 | 独立进程加载模型 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读生成循环与训练目标的关系 |
| 2h | 实现 generate.py 与自回归生成函数 |
| 1h | 保存加载对照及固定样例 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `scripts/generate.py`，接收 `--checkpoint`、`--prompt`、`--max-new-tokens`、`--seed` 和 `--output`。从 checkpoint 重建模型、词表和 context，不手工另建不一致映射。`--output` 指定本次 JSONL 样例文件，每次完整输出追加为一条记录。

2. 生成时 eval 并关闭梯度，每轮只保留最后 context 个输入 ID，取最后位置 logits，经 softmax 后采样一个 token，拼回序列；本周使用 temperature=1 的基础采样。

3. 固定 prompt（例如训练词表内的角色名）、seed=42、max_new_tokens=200，分别用 Bigram 和 GPT 生成，保存 `samples.jsonl` 的全部原始输出和设置。

4. 检查保存前与重新加载后，在相同环境、prompt、seed 和采样设置下可复现；解释一次 forward 的并行训练与逐 token 生成的区别。

### 分步操作与自查

#### 第一步：只取最后位置的分布

创建 `scripts/generate.py`，顶层导入 argparse、json、Path、torch、CharTokenizer、BigramLanguageModel 和 MiniGPT。在 main 中解析任务概览列出的五个参数，按第二步从 checkpoint 重建对象，再调用本步生成循环和第三步的 JSONL 保存逻辑；文件末尾添加入口保护。只导入脚本不会加载权重或自动生成文本。

生成给定前缀后的下一个 token 时，模型返回 `[B,T,V]`，使用 `logits[:, -1, :]` 得到 `[B,V]`；其他位置对应前缀内部的预测。

先从 checkpoint 还原模型、tokenizer 和 context_length，按第二步说明在加载后重新设置生成 seed。然后把非空 prompt 编码成 `[1,T]` 的 long Tensor；下面片段先放在 generate.py 的 main 中，args 已由 parse_args 得到：

```python
prompt_ids = tokenizer.encode(args.prompt)
if not prompt_ids:
    raise ValueError("prompt 必须非空")
token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
model.eval()
with torch.no_grad():
    for _ in range(args.max_new_tokens):
        context_ids = token_ids[:, -context_length:]
        logits = model(context_ids)
        probabilities = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_id), dim=1)
```

参数解析时 max-new-tokens 与 seed 用 type=int，checkpoint 和 output 用 Path，prompt 为字符串；max-new-tokens 要为非负数。每次追加一个 ID 后再次前向，重复 max_new_tokens 次。上下文裁剪影响模型每次可见输入，不要把累积输出文本也截掉。循环后用 `all_ids = token_ids[0].cpu().tolist()` 得到完整 ID 列表，`new_ids = all_ids[len(prompt_ids):]` 得到新增列表，再用 `tokenizer.decode(all_ids)` 得到完整文本；这些实际值用于后面的 JSONL 记录。

先用 3 个已知 prompt ID、max_new_tokens=5 检查最终保留 8 个 ID，前 3 个仍与 prompt 相同；再构造长于 context 的前缀，检查每次送入 forward 的长度不超过 context，累积结果仍持续增长。示例 prompt 必须非空；空 prompt 在本轮明确提示需要输入，避免索引最后位置失败。

#### 第二步：用 checkpoint 还原完整配套信息

新进程读取结构、context 与 tokenizer，创建模型并加载；固定 seed，在 eval/no_grad 下生成。脚本不应依赖训练程序内存中的对象。

generate.py 的 main 按以下顺序接入：load_checkpoint(args.checkpoint) → 核对 tokenizer 文件 hash → CharTokenizer.load → 核对 tokenizer.vocab_size 与 checkpoint.vocab_size → 根据 model_config.name 新建 Bigram 或 MiniGPT → 严格 load_state_dict → 迁移设备并 eval。MiniGPT 的 heads 仍映射为构造参数 num_heads；context_length 从 checkpoint 读取，不在生成脚本另写 128。设备可沿用第二周的自动选择，打印实际设备；旧设备保存在 checkpoint 中不要求本次必须相同。

若后续把 main 中的循环抽成函数，可约定 `generate_ids(model, prompt_ids, context_length, max_new_tokens, device)`，返回完整 token ID 列表；抽取时将循环里的 args 字段改为函数参数，编码与参数解析仍由 main 完成。下方 JSONL 记录还要写实际 device / dtype、prompt 的 OOV 数；第七周再将加载替换为按 tokenizer 类型分派的入口。

重建和加载模型会涉及初始化，因此生成复现用的 seed 在模型加载完成后、采样开始前重新设置。保存前与加载后的两次生成都从相同 prompt 和 seed 开始，比较完整新增 token ID 列表完全一致；不能只对比解码后看起来相似的文字。

<a id="jsonl-output"></a>
#### 第三步：保存可解释的样例

samples.jsonl 每行记录 model、checkpoint、tokenizer_hash、prompt、seed、生成设置和原始文本。角色名等短 prompt 要先确认属于字符词表。

JSONL 是每行一个完整 JSON 对象。先构造本次实际结果字典 record，文本中的换行由 json.dumps 转义；一条记录只占文件的一行：

```python
import json
from pathlib import Path

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("a", encoding="utf-8") as file:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
```

读取时逐行 `json.loads(line)`，不要把整个 JSONL 当成一个 JSON 字典读取。record 同时保存 prompt IDs、新增 IDs、实际新增数量和设置，核对与终端一致；重复运行会追加独立记录。第七周和第八周的原始输出沿用这一格式。

#### 第四步：解释生成结果

比较局部拼写、角色格式、重复和连贯性。Bigram/GPT 都是小语料教学模型，样例用于理解训练目标与架构，不将少数顺畅句子当整体能力证明。

### 运行与验收

```bash
python scripts/generate.py --checkpoint outputs/bigram_char/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --output experiments/mini_gpt/week6/samples.jsonl
python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --output experiments/mini_gpt/week6/samples.jsonl
```

打印新增 token 数与完整文本，文件保留两组各自的模型、tokenizer 和设置。位置索引越界时，检查是否每次只把最后 context_length 个 ID 输入模型；checkpoint 的 V 与词表不匹配时，先修复配套信息，不跳过输出层权重。

- [ ] 生成长度、上下文截断与 token dtype 正确。
- [ ] 导入模型不运行训练，独立脚本可加载生成。
- [ ] 样例标注真实模型、tokenizer、prompt、seed 与设置。

<a id="day-6"></a>
## Day 6：语言模型比较与核心流程重写

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度评估说明（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | PPL 定义与 tokenizer 对指标的影响 | 理解下周 BPE 比较边界 |
| [总计划语言模型验收](StudyPlan.md#acceptance) | 同 tokenizer 的基线要求 | 整理本周验证结果 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 核对曲线、tokenizer 与配置 |
| 1.5h | 分析验证差异与生成样例 |
| 1h | 有限参考重写前向/生成核心 |
| 0.5h | 更新本周 README |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 保存 `week6/results.csv`：model、tokenizer_hash、split=val、val_loss、参数量、steps、耗时；把两组验证 loss 与生成样例放在同一口径下解释。

2. 确认 GPT 验证 loss 优于同字符 tokenizer 的 Bigram；若未达到，保留真实结果并顺延该关卡，说明尝试和下一步。

3. 有限参考下重写错位样本、下一 token 交叉熵和逐步生成核心，解释 causal Mask 为什么必要。

4. 更新本周 README，链接曲线、样例、split manifest 和 tokenizer；原文测试集继续保留，Temperature、Top-k 和 BPE 按第七周安排。

### 分步操作与自查

#### 第一步：核对最终验证表

将两组真实结果整理为 `experiments/mini_gpt/week6/results.csv`，用第一周 CSV 写法保存两行；字段包含 model、tokenizer_hash、split=val、best_step、val_loss、val_target_tokens、实际完成 steps、params、train_seconds、config、checkpoint。最佳验证值与目标数读取各自 best 对应的记录，不从测试集取值，也不分别挑选不同验证时点的数据。

model、tokenizer_hash、val_loss、steps、params 和时间都对应实际运行。字符两组 tokenizer_hash 必须一致，才有直接比较 token loss 的前提。

若报告 PPL，使用相同口径 loss 的 exp；本周并不要求额外实现复杂 PPL 评估，先保证 NLL 与有效 token 数正确。

#### 第二步：解释模型实际学到了什么

把 loss 差异、生成例子与架构联系起来：Bigram 只依赖当前 token，GPT 使用因果前缀。也说明小数据、单种子与有限预算的限制。

#### 第三步：有限参考写出三个核心片段

写 T+1 片段如何得到 x/y、CrossEntropy 的 reshape、逐 token 生成循环。先标 shape，再写操作，不需要再复制整个训练工程。

用草稿实际生成 `[0,1,2]→[1,2,3]` 的错位对，检查展平后的 loss 为有限标量；再让草稿生成循环追加 5 个 token，确认输入只截断上下文、完整输出长度仍增加 5。已有模型测试之外，这三段重写也要被调用核对。

#### 第四步：为 BPE 和续训保留边界

保存原文划分、字符映射、模型配置和 best/last。测试原文继续保留；下周变更 tokenizer 时创建新模型并单独报告指标，不直接把两个 token loss 作大小排序。

### 运行与验收

```bash
python -m pytest tests/test_language_model.py -q
```

对照两个 best 指标、history、曲线和 samples，填写 `experiments/mini_gpt/week6/results.csv`。今天以已有验证结果复盘，不为整理表格重新训练，也不评估保留测试原文。复查模型时使用 Day 4 已编写的加载与验证函数核对同一 best。

- [ ] 同 tokenizer 的 GPT/Bigram 对照有真实指标与失败分析。
- [ ] 能解释训练目标、验证统计和自回归生成。
- [ ] 下一周可复用相同原文划分、模型结构与字符基线。

## 本周结束检查

- [ ] 原文划分先于窗口构建，词表仅来自训练数据。
- [ ] Bigram 与 GPT 在相同字符 tokenizer 下比较，GPT 验证 loss 更低。
- [ ] 因果信息隔离、保存加载与生成复现检查通过。
- [ ] 配置、tokenizer、曲线、样例和结果口径在本周 README 中齐全。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第七周](week7.md)。

