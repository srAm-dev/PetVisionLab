# 第六周：字符级 Bigram 与 Mini-GPT

从下一字符预测开始，先写 Bigram，再组装 Mini-GPT。两者使用同一字符词表和固定文本划分，比较验证 loss 与生成样例。

对应[总计划第 6 周](StudyPlan.md#week-6)。安排为 6 个学习日，每天 4 小时任务和 1 小时机动，共 30 小时；第 7 天休息。Day 1 可从任意星期开始，卡住时先补当前步骤，再顺延后面的任务。

环境：本地 Conda `petvision`、RTX 4060；起步 2 层、C=128、4 heads、context=128、batch=16，FP32，按实际显存与耗时调整。

## 开始前

第四周 causal Mask 已通过测试；能独立写训练、验证和 checkpoint 流程。保留第五周的视觉结果。

按当天的阅读范围、操作步骤和检查顺序学习。通用的路径、代码片段和记录约定见[执行说明](README.md#working-rules)；实际进度、命令和结果写在[本周实验记录](../experiments/mini_gpt/week6/README.md)。

更新日期：2026-11-10。

## 本周文件

按注释中的学习日逐步创建或修改文件；已有文件继续使用，结果在运行后保存。

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

## 环境检查

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

确认解释器属于 `petvision`，`petvision.__file__` 指向当前仓库的 `src/petvision/`。导入失败时先在根目录运行 `python -m pip install -e . --no-deps`；依赖或 CUDA 问题按[第二周环境排查](week2.md#environment)处理。

复用已经通过检查的 Attention、训练和保存加载方法，新增文本数据与模型。本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

<a id="lm-shapes"></a>
本周模型在每个文本位置预测下一个 token，为该位置输出词表中各项的分数。

| 符号 / 名称 | 含义 | 例子或对应配置 |
| --- | --- | --- |
| token / ID / tokenizer | 文本处理单位 / 该单位的整数编号 / 编解码工具 | 本周一个已知字符对应一个 token；ID 不表示数值大小或语义相近 |
| V / vocab_size | 词表大小，含 UNK | 若已知字符有 65 种，加一个 UNK 后 V=66，合法 ID 为 0..65 |
| B / batch_size | 同时处理的文本片段数 | 16 |
| T / context_length | 本次输入长度 / 模型允许的最大上下文长度 | 训练通常 T=128；尾窗和生成时 T 可更小 |
| C / embed_dim | 每个 token 对应的浮点特征向量长度 | 128，不是词表大小 |
| N | 当前 split 编码后的 token 总数 | 用于采样边界和 N-1 个验证目标的计数 |
| NLL / PPL | 负对数似然 / 困惑度 | NLL 是正确 token 概率的负自然对数；平均 NLL 的 exp 为本口径的 PPL |

文本采用作者仓库 Tiny Shakespeare，下载后记录实际来源 URL、文件 SHA256 与解码/换行处理方式。按 Unicode 字符位置顺序切成 80%/10%/10% 的 train/val/test，先保存原文，再分别编码和造窗口；窗口不跨边界。字符词表只从 train 建立，额外保留 `<unk>` ID；验证或 prompt 的未见字符映射到 UNK（未知字符占位 ID），并报告 OOV（未收录字符）的数量，已知字符的 encode/decode 应往返一致。

GPT 的正式结构以 [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) 的 GPT-2 风格为参考：pre-LN、GELU、可学习位置嵌入、embedding dropout、最终 LayerNorm，以及 token embedding / 无 bias 输出头的权重共享。Attention 与 LayerNorm 沿用第四周基础模块，GPTBlock 在本周文件中定义。数据与小预算是项目配置，模型规模不复制作者的大模型默认值。

Bigram 与 GPT 使用相同 tokenizer 文件、数据划分、context=128、batch=16、seed=42、AdamW lr=3e-4、weight_decay=0.01、2000 steps、每 200 步验证。GPT 为 2 层、C=128、4 heads、FFN=512、dropout=0.1；Bigram 为 vocab×vocab 转移 logits 的 Embedding。NLL 指负对数似然，即正确目标 token 的概率取负自然对数。验证在相同原文上按固定顺序统计有效目标 token 的总 NLL/总数，关闭 dropout；按[验证分窗规则](#eval-windows)覆盖 N-1 个目标，尾窗单独评估并记录计数。测试原文留到第七周配置冻结后使用。预算不足时先短运行测时，再同时固定两组实际预算。

## 六天安排

| 学习日 | 修改哪些文件、按什么顺序做 | 做完应看到什么 |
| --- | --- | --- |
| [Day 1](#day-1) | 完成 `data/text.py` 和字符 tokenizer，再编写 `scripts/prepare_text.py` | 原文按 80/10/10 划分；保存 manifest 和词表；统计 OOV |
| [Day 2](#day-2) | 完成窗口、Bigram、`lm_trainer.py` 和语言模型测试，再接配置与 `scripts/train_gpt.py` | 输入目标错开一位；验证恰好统计 N-1 个目标；先只做小批检查 |
| [Day 3](#day-3) | 创建 `models/llm/mini_gpt.py` 及 GPT 配置 | 前缀不受未来扰动；[B,T] 输出 [B,T,V]；短文本可拟合 |
| [Day 4](#day-4) | 补齐 `scripts/train_gpt.py` 的验证、日志与 checkpoint | 两组正式训练完成相同预算，按 val loss 选 best 并保存曲线 |
| [Day 5](#day-5) | 创建 `scripts/generate.py`，保存 `samples.jsonl` | 只取最后位置分布逐 token 生成；加载前后固定 seed 的新增 IDs 一致 |
| [Day 6](#day-6) | 整理本周 `results.csv`、周总结，重写采样与 loss 片段 | 同 tokenizer 的验证结果可比较；测试文本未参与选参 |

<a id="day-1"></a>

## Day 1：原文划分、字符词表与数据来源

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [D2L 文本预处理](https://zh.d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html) / [语言模型与数据集](https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html)（主讲解） | 词表、字符与下一个 token 目标；保留项目的原文和划分规则 | 明确文本、ID、窗口与预测目标 |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | Tiny Shakespeare 的数据与字符编码部分 | 对照作者使用的数据入口 |
| [Tiny Shakespeare 原文](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) | 查看格式、换行和角色名，记录下载来源 | 先划分原文再编码 |
| [Unicode 指南](https://docs.python.org/zh-cn/3.12/howto/unicode.html)、[hashlib](https://docs.python.org/zh-cn/3.12/library/hashlib.html)（官方中文） | UTF-8 编解码、sha256 与 hexdigest；JSON 读写复用第二周 | 区分原始 bytes、实际使用文本及各自 hash |

### 操作步骤

划分记录保存到 `experiments/mini_gpt/week6/split_manifest.json`，第七周继续使用同一份文件；保留原文空白和换行。

原文是字符串；字符 tokenizer 把每个已知字符映射为整数 ID；模型接收 ID，并不会直接读取 Python 字符串。空格和换行也是训练内容。

```text
文本： 'ab\na'
映射： {'\n':0, 'a':1, 'b':2}
编码： [1,2,0,1]
解码： 'ab\na'
```

实际映射只来自训练文本，排序后保存，不能每次运行重新按集合遍历顺序编号。另保留 UNK 处理未见字符。

#### 第一步：在 data/text.py 完成 split_text

创建 `src/petvision/data/text.py`，将按字符边界切分的逻辑封装为 `split_text(text)`，返回 train_text、val_text、test_text 三个字符串；明天继续在同一模块加入窗口函数。

按字符位置切 train/val/test。不能先对整个原文生成大量重叠窗口，再随机把窗口分给 train/val，那样相邻文本可能泄漏。第三步再由脚本保存边界与 hash。

在 split_text 中计算 L、train_end=int(0.8*L)、val_end=int(0.9*L)，按边界切出三个字符串并返回。这里完成切分函数；文本写入、hash 和 manifest 由第三步的数据准备脚本处理。

#### 第二步：在 tokenizer.py 完成 CharTokenizer 的编解码与保存加载

创建 `src/petvision/models/llm/` 和空的 `__init__.py`，再创建 `src/petvision/models/llm/tokenizer.py`，将下面的类与 JSON 读写实现写进该文件。

```python
class CharTokenizer:
    def __init__(self, chars): ...
    def encode(self, text): ...
    def decode(self, ids): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

构造函数接收已经排序的训练字符列表 chars，建立字符→ID 和 ID→字符的对应关系，同时保存 self.chars、self.unk_id 和 self.vocab_size。

load 是类方法：它从 JSON 读回 chars，用 `cls(chars)` 创建同一种 tokenizer，再核对 version、unk_id，返回这个对象。重新加载时不再读取文本重建词表。已知字符必须往返一致；UNK 映射无法恢复原字符，需要报告 OOV，不把这种输入当成已知字符往返检查。

为避免保存后 ID 改变，本轮可约定 `chars=sorted(set(train_text))`，已知字符 ID 为 `0..len(chars)-1`，UNK ID 为 `len(chars)`，实际 V 为 `len(chars)+1`。JSON 保存 `type="char"、version=1、chars、unk_id`，加载后按列表顺序还原映射，decode 遇到 UNK 可输出 `<unk>` 占位符。用训练文本 `ab\na` 检查已知编码 `[1,2,0,1]`，未见字符 `?` 编为 3，OOV=1；该未知字符的原值不能由 3 恢复。

把上述接口逐个补齐，每完成一个就用这个短串检查：

1. 构造函数按 chars 的顺序编号，建立两个方向的映射，再设置 unk_id 和 vocab_size；打印短串例子的映射，确认换行占一个独立 ID。
2. encode 从左到右遍历字符串，每次查一个字符；存在就追加对应 ID，不存在就追加 unk_id。不要删掉空格或换行，也不要在这里临时扩充词表。
3. decode 从左到右遍历 ID，已知 ID 查回字符，unk_id 转为约定占位符，最后连接成一个字符串；超出合法范围的 ID 明确报错。
4. save 将格式版本、按顺序保存的 chars 和 unk_id 写入 JSON。文件里保留字符本身，由 JSON 库处理换行转义，不手工拆分行。
5. load 先核对 type/version，再用保存的 chars 重建对象，核对 unk_id；不重新扫描 train.txt，也不改变列表顺序。
6. 先检查已知短串的编码与往返，再单独加入 `?` 检查 UNK。最后对保存前后的两个对象编码同一输入，要求 ID 列表逐项相同。

#### 第三步：完成 prepare_text.py，保存数据、词表并核对 OOV

创建 `scripts/prepare_text.py`，从 `petvision.data.text` 导入 split_text，在 main 中读取原文、调用切分、保存文本和 manifest，末尾添加入口保护。下载与文件写入由脚本执行，导入数据模块不自动下载。

在 prepare_text.py 顶层用 `from petvision.models.llm.tokenizer import CharTokenizer` 导入，数据准备脚本只调用它，不再定义另一套字符映射。

先用浏览器从原文链接保存到约定路径，或在 prepare_text 中使用标准库下载；两种方式都读取实际落盘 bytes。hash 的最小用法如下：

```python
import hashlib
from pathlib import Path

raw_bytes = Path("datasets/raw/tinyshakespeare/input.txt").read_bytes()
raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
text = raw_bytes.decode("utf-8")
used_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
```

设完整字符串长度为 L，令 `train_end=int(0.8*L)`、`val_end=int(0.9*L)`，依次保存 `text[:train_end]`、`text[train_end:val_end]`、`text[val_end:]` 到 `datasets/processed/tinyshakespeare/{train,val,test}.txt`。三个区间首尾相接，应覆盖全部字符。

manifest 至少记录 source_url、原始文件 sha256、encoding、换行处理方式，以及三个 split 的路径、`[start,end)` 区间和各自 hash。下载时保留原始 bytes 的 hash；解码、处理换行后再记录实际使用文本的 hash。再次运行优先核对已有文件，避免悄悄覆盖第七周正在使用的数据。

记录下载文件 hash 和读取/换行规范化方式；同一来源 URL 未来内容变化时，hash 能帮助确认是否还是同一份数据。

本轮可直接保留原换行，三个 split 用 `Path.write_bytes(part_text.encode("utf-8"))` 写入，父目录先创建。断言三个字符串按顺序拼接等于 text，再计算各 split 的 UTF-8 bytes hash；manifest 的原文件 hash 与实际文本 hash 分别命名。后续读取已有文件时检查 hash 和区间，不能只看文件名相同。

在 `scripts/prepare_text.py::main()` 写出 train.txt / val.txt / test.txt 后，将来源、边界和 hash 组织成 manifest 字典；先创建 `experiments/mini_gpt/week6/`，再用 `Path(.../split_manifest.json).open("w",encoding="utf-8")` 打开文件，调用 `json.dump(manifest,file,ensure_ascii=False,indent=2)` 保存。记录前面计算出的实际字符数和各 split 长度，保存后读回核对三个区间连续、总长为 L。

prepare_text.py 在完成原文划分后调用 `CharTokenizer(sorted(set(train_text)))`，再保存到 `experiments/mini_gpt/week6/char_tokenizer.json`。

重新加载 JSON，编码同一短文本，ID 序列应一致；保存词表大小与文件 hash，后续 checkpoint 按这个标识加载。

将这些检查接到 `scripts/prepare_text.py` 的保存操作之后，打印已知短文本及往返结果，并分别统计 val/test 中不属于训练字符集合的字符数量；不扩充词表。再次运行时核对已有 split 与 tokenizer，没有条件变化则复用，不静默覆盖跨周使用的文件。

### 运行与自查

```bash
python scripts/prepare_text.py
```

应打印来源、原文长度、三个区间、实际词表大小和 OOV 数，保存三个文本、split manifest 与 `char_tokenizer.json`。重新加载 tokenizer 后，对包含换行的已知字符例子检查往返；val 出现 UNK 时保留数量，不把验证字符加回训练词表。

- [ ] 原文先划分，所有后续模型和 tokenizer 复用相同边界。
- [ ] 词表来源仅为训练文本，OOV 策略明确。
- [ ] tokenizer 可保存加载，已知字符往返不改变空白。

<a id="day-2"></a>

## Day 2：下一 token 窗口与 Bigram 基线

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [D2L 8.3 语言模型和数据集](https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html)（主概念材料） | 自回归概率、序列采样、X/Y 错位 | 先手算下一个 token 目标；正式窗口按本周固定规则 |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 字符编码、batch 采样与 Bigram 部分，按主题观看 | 实现输入与目标错位 |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | 对应课程最小 Bigram 实现 | 理解 vocab×vocab logits |

### 操作步骤

#### 第一步：在 data/text.py 完成训练采样与完整验证分窗

继续修改 `src/petvision/data/text.py`，今天新增 sample_batch 和本步的 iter_eval_windows。

设 ID 为 `[0,1,2,3,4]`，T=3，起点 0：

```text
取出的 T+1 个 token：[0,1,2,3]
x：                 [0,1,2]
y：                 [1,2,3]
位置 0 用 token 0 预测 token 1
位置 1 用 token 1 预测 token 2
```

目标比输入向后错开一位；整个片段应位于当前 split 内。

长度为 N 的 token 序列中，若每次取 T+1 个 token，起点为 `0..N-T-1`；`torch.randint` 的 high 为 `N-T`，因为上界不包含在结果内。例如 N=5、T=3 时只能从 0 或 1 开始。N 小于 T+1 时明确提示数据不足，不从另一个 split 借 token。

在 data/text.py 实现 `sample_batch(token_ids, context_length, batch_size, generator, device)`。token_ids 为当前 split 的一维 CPU long Tensor；函数用 `torch.randint(0, N-T, (batch_size,), generator=generator)` 产生起点，将每个 `token_ids[start:start+T]` 与 `token_ids[start+1:start+T+1]` 分别用 `torch.stack` 拼为 x / y，再移到 device 后返回这两个 Tensor。stack 是在已有样本外新增一个 batch 维，可查 [torch.stack 官方接口](https://docs.pytorch.org/docs/stable/generated/torch.stack.html)。

调用者在训练开始前创建一次独立 CPU `torch.Generator().manual_seed(42)` 并持续传入，不能每批重新创建或重设种子。输入与 target 都为 long，模型参数是浮点数。本周训练直接从 token 序列取窗口，不需要套用图像 DataLoader。

<a id="eval-windows"></a>

##### iter_eval_windows

训练按随机窗口采样；验证使用确定性的分窗，不抽取随机验证小批替代完整统计。本周不引入 BOS：当前 split 编码后有 N 个 token，索引 0 没有本 split 内的前驱，不计为目标；索引 1..N-1 各预测一次，目标总数应为 N-1。

BOS 是 beginning-of-sequence，即额外添加的“序列开始”token。本轮不加它，因此第一个真实 token 只作为输入，不能假设前面还有一个可预测它的位置。有效目标数指实际参与交叉熵统计的位置数，不是窗口数，也不是原文文件的字节数。

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

评估时先将 x、y 作为 long Tensor 移到模型设备，分别补 batch 维为 `[1,t]`；模型返回 `[1,t,V]`。t 可以小于 context，尤其尾窗不能强行 reshape 成固定长度。模型切 eval 并关闭梯度，每窗用 `cross_entropy(..., reduction="sum")` 得到 NLL 总和，累加 `y.numel()` 个目标，最后计算 `total_nll / total_target_tokens`。

上例计数应为 `3+3+1=7=N-1`。若三个窗口的 mean loss 分别为 1、2、4，整体 loss 为 `(3*1+3*2+1*4)/7=13/7`，不能对三个 mean 直接平均。补到现有语言模型检查中：普通尾窗、刚好整除、N=2、不同 context 均满足目标索引恰好覆盖 1..N-1。

该规则会在窗口边界重置模型可见上下文；它是本轮约定的固定评估方式，Bigram/GPT 必须一致，不把它和使用滑动长前缀的评估结果混在一列。每个 split 单独处理，不从 train 借前缀给 val，也不跨入 test。

#### 第二步：在 bigram.py 完成 BigramLanguageModel

创建 `src/petvision/models/llm/bigram.py`，顶层写 `from torch import nn`，在其中实现下述 BigramLanguageModel。

Embedding(V,V) 可视为一张转移 logits 表，当前 token ID 取出一行，决定下一 token 的分布。它不利用更早的上下文；GPT 将能利用前缀信息。参数量为 V²。在 bigram.py 定义 `BigramLanguageModel(vocab_size)`，构造函数创建 `nn.Embedding(vocab_size, vocab_size)`，forward(token_ids) 返回查表结果；token_ids 必须是 long Tensor。

Embedding 是可学习的查找表：第一个 V 是“有多少行可供 ID 索引”，第二个 V 是“每行有多少个数”。假设 V=4，输入 ID=2 就取表的第 2 行，得到对下一 token 为 0、1、2、3 的四个 logits；不是把数字 2 乘上权重。输入 `[B,T]` 的每个 ID 都取一行，结果为 `[B,T,V]`。这里特意让每行宽度等于 V，以便直接作为分类分数；GPT 的 token embedding 则为 Embedding(V,C)，先取 C 维特征，再经网络输出 V 维分数。

#### 第三步：在 lm_trainer.py 完成 train_step 与 evaluate_lm

创建 `src/petvision/training/lm_trainer.py`，导入时不启动训练。先按下面的形状组织交叉熵，再完成两个训练 / 评估函数。

```text
输入 x / 目标 y      [B,T]，整数 ID
Bigram logits        [B,T,V]
loss 输入 logits     [B*T,V]
loss 输入 target     [B*T]
```

例如 B=2、T=3、V=4：logits 为 `[2,3,4]`，targets 为 `[2,3]`；展平后分别为 `[6,4]` 和 `[6]`，相当于一次评估六个“四选一”问题。`reshape(-1,V)` 只合并 batch 和位置两轴，不合并词表轴；`targets.reshape(-1)` 使用相同顺序，保证每个位置与自己的下一 token 标签对应。

把自回归概率也对应到这个例子：一个 T+1 片段训练的条件概率是各位置下一 token 概率的乘积；取负对数后变为求和，批次 loss 再除以 `B*T`。Bigram 每一项只依赖当前 ID，GPT 每一项可依赖当前可见前缀。用 `[0,1,2,3]` 逐项指出「0→1、1→2、2→3」，解释为什么目标不能与当前位置输入完全相同。

在 lm_trainer.py 实现 `train_step(model, x, y, optimizer)`：切 train、清梯度、前向、将 logits / y 展平计算交叉熵、backward、step，返回本批平均 loss 和目标数。`evaluate_lm(model, token_ids, context_length, device)` 切 eval/no_grad，完整累加所有窗口的 NLL 与目标数，返回 loss 和 n_target_tokens，再恢复原模式。两个函数的输入来自不同 split，不能在内部重新随机划分。

两个函数统一返回 `{"loss": float值, "n_target_tokens": int值}`，键名固定；train_step 的数量为 y.numel()，evaluate_lm 为完整分窗的累计目标数。lm_trainer.py 顶层导入 torch、`torch.nn.functional as F` 和 iter_eval_windows；训练 loss 使用 `F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))`，验证使用相同展平方式但 reduction="sum"。evaluate_lm 开始先保存 `was_training = model.training`，在 try 中执行 eval/no_grad，在 finally 中 `model.train(was_training)`，确保退出时恢复原模式。

#### 第四步：编写 test_language_model.py 的窗口、模型和评估检查

创建 `tests/test_language_model.py`，顶层导入 torch、pytest、sample_batch 和 iter_eval_windows。新增 `test_sample_batch_shift` 检查 x/y 错开一位；`test_sample_batch_split_bounds` 检查样本不越过当前 split、数据不足时抛出 ValueError；`test_eval_window_coverage` 将下方 N=8、T=3 的各窗口目标拼接，断言恰好覆盖 ID 1-7 且无重复。

在 `tests/test_language_model.py` 顶层从 `petvision.models.llm.bigram` 导入 BigramLanguageModel，新增 `test_bigram_forward_backward`：用 V=8、B=1/2、T=4 的合法 long ID 检查 `[B,T,V]` 输出，展平后计算交叉熵并 backward，断言 Embedding 参数梯度非 None 且有限。

在本日 test_bigram_forward_backward 中实际执行这两次 reshape，把原始 logits 和展平后的目标传给 CrossEntropyLoss，检查 loss 为有限标量后反向。每个位置是一个 V 分类问题，目标 ID 范围必须在 0..V-1；交叉熵直接接收 logits。

修改 `tests/test_language_model.py::test_bigram_forward_backward`：创建 AdamW 后先 clone Embedding 参数，再计算 loss、清梯度、backward、optimizer.step，断言至少一个参数值发生变化。另新增 `test_lm_evaluation_preserves_generator`：创建独立训练 Generator，保存其 get_state().clone()，对固定 val_ids 调用 evaluate_lm，再用 torch.equal 比较 Generator 状态完全相同，同时核对返回的目标数等于 len(val_ids)-1。

<a id="text-inputs"></a>

#### 第五步：创建 bigram_char.yaml

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

#### 第六步：完成 train_gpt.py 的配置读取与小批调用

创建 `scripts/train_gpt.py`。在 train_gpt.py 顶层导入 argparse、yaml 和 pathlib.Path，定义 parse_args(argv=None)，用 argparse.ArgumentParser 添加必填的 `--config`（type=Path）；main() 调用 parse_args，再以 UTF-8 打开配置文件，用 yaml.safe_load 得到 cfg，随后读数据、创建对象和调度训练。文件末尾添加 `if __name__ == "__main__": main()` 入口保护。

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

读取前用 split_manifest 核对文件 hash；按 bytes 解码可保持之前约定的换行。今天测试文本不进入训练 / 验证循环。在 train_gpt.py 顶层导入 `petvision.utils.seed.set_seed`；main() 将 device=auto 解析为 CUDA 可用时 cuda、否则 cpu，并用 torch.device 创建对象，先调用 set_seed(cfg["training"]["seed"])，再新建 `BigramLanguageModel(tokenizer.vocab_size).to(device)` 和 AdamW。一次更新先调用 sample_batch(train_ids, ...)，再将得到的 x / y 交给 train_step；验证把 val_ids 交给 evaluate_lm，按前述 iter_eval_windows 遍历。

入口还需校验 training.steps、eval_interval、model.context_length、data.batch_size 为正整数；目前 name 仅支持已实现模型，optimizer 仅支持 adamw，dtype 仅支持 float32。data.name=tinyshakespeare 对应本周保存的原文文件；其他值明确报错。这样修改配置不会被代码静默忽略。

词表大小从 tokenizer.vocab_size 实际读取；`model.context_length` 控制窗口长度，Bigram 本身仍只依赖当前位置。一次 step 指一个训练 batch 的参数更新，2000 steps 不是遍历完整语料 2000 次。今天先实现参数解析与小批更新，正式结果待 Day 4 写出。

### 运行与自查

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

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)（正式结构参考） | MLP、Block、GPT.__init__、forward 和初始化；先忽略优化 / 预训练权重导入部分 | 按下方步骤组织小 GPT |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（教学视频） | 从 Bigram 逐步加入 Attention、位置嵌入和残差 | 理解构建过程；最终 GELU 与权重共享按上述作者实现 |
| [第四周指南](week4.md#block-conventions) | Attention、LayerNorm、causal Mask 与未来隔离 | 复用已验证的基础模块 |

### 操作步骤

#### 第一步：在 mini_gpt.py 完成 GPTBlock、MiniGPT 构造与 forward

创建 `src/petvision/models/llm/mini_gpt.py`，顶层导入 torch、nn、math，以及第四周的 MultiHeadSelfAttention 和自写 LayerNorm。第四周 EncoderBlock 是教材的 post-LN＋ReLU，GPT 在这个文件中定义自己的 pre-LN Block。

先定义 `GPTBlock(embed_dim, num_heads, ffn_hidden_dim, dropout=0.0)`：

1. 创建独立的 norm1、norm2，eps=1e-5；两个归一化均保留 gamma、beta。
2. 创建 MultiHeadSelfAttention，显式设 bias=True。它使用三套 Q/K/V Linear，作者的 c_attn 将它们合为一个 Linear；计算等价，本项目继续分别定义 q_proj / k_proj / v_proj。后面的 test_gptblock_reference 将三套 weight 和 bias 按 Q/K/V 顺序沿 dim=0 拼接，复制到参考 Attention 的 in_proj_weight / in_proj_bias，核对这种对应关系。
3. Attention 的 softmax 权重 dropout 保留在注意力模块内部；GPTBlock 再注册一次投影输出 dropout，对应作者 resid_dropout。第一条残差为 `x = x + attn_output_dropout(attention(norm1(x), mask))`。
4. FFN 按 Linear(C,hidden_dim) → GELU → Linear(hidden_dim,C) → Dropout 组织，两个 Linear 使用 bias=True；这里按 nanoGPT 在第二个 Linear 后做一次 dropout，GELU 后不另加隐藏层 dropout。第二条残差为 `x = x + ffn(norm2(x))`。
5. GPTBlock.forward 接收 x、attn_mask、key_padding_mask=None，将 Mask 传给 Attention，只返回新的 x。

再定义 `MiniGPT(vocab_size, context_length, embed_dim, depth, num_heads, ffn_hidden_dim, dropout)`，按下面顺序建立对象：

1. 从 tokenizer 取实际 V，记录 context_length 与 C，检查 C 能被 num_heads 整除。YAML 的 heads 由入口显式传给 num_heads。
2. 注册 `token_embedding = nn.Embedding(V,C)` 和 `position_embedding = nn.Embedding(context_length,C)`，再注册 embedding_dropout。两种嵌入相加后做一次 dropout；GPT 不使用第四周文本 Encoder 的 sqrt(C) 缩放或正弦位置表。
3. 按 depth 分别新建 GPTBlock，放入 ModuleList，各层参数独立。本周首轮 C=128、FFN=512、depth=2、heads=4。
4. 创建最终 LayerNorm(C)，再创建 `head = nn.Linear(C,V,bias=False)`；在统一初始化之前令 `self.head.weight = self.token_embedding.weight`，共享同一个 Parameter。两者 shape 都为 `[V,C]`：输入时按 token ID 取行，输出时用这些行与隐藏状态计算词表 logits。位置嵌入单独保留。
5. 按作者初始化：Embedding、Linear 的 weight 用均值 0、std=0.02 的正态分布，存在的 Linear bias 置零；LayerNorm 的 gamma/beta 保持 1/0。可在模型中定义初始化方法并用 apply 遍历。随后把各 GPTBlock 的 Attention 输出投影和 FFN 第二个 Linear 的 weight 标准差调整为 `0.02 / sqrt(2*depth)`，对应作者残差投影的初始化。
6. 用 `model.head.weight is model.token_embedding.weight` 检查共享关系；named_parameters 默认对同一 Parameter 去重，因此不要求这份共享权重被列出两次。保存加载后也检查共享关系及固定输入输出；AdamW 直接使用 model.parameters()，不手动将共享权重重复加入参数组。

```text
token IDs                    [B,T]
token + learned position     [B,T,C]
embedding dropout            [B,T,C]
depth 个 causal GPTBlock     [B,T,C]
最终 LayerNorm               [B,T,C]
共享输入权重的无 bias head   [B,T,V]
```

位置 ID 为 0..T-1，输入 T 不超过 context_length；head 的 V 来自 tokenizer，不与上下文长度混用。训练 loss 继续由训练器根据 logits 和 targets 计算。

```python
def forward(self, token_ids):
    # 返回 logits: [B,T,V]
    ...
```

在训练器中根据 logits 和 targets 计算 loss，这样同一模型也能只前向做生成。GPTBlock 将第四周注意力核心的布尔 Mask 接口传下去；本轮使用明确的上三角 Mask，不另加 causal=True 参数。在 MiniGPT.forward 中，取 B、T = token_ids.shape，检查长度后创建：

```python
positions = torch.arange(T, device=token_ids.device)
x = self.embedding_dropout(
    self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
)
causal_mask = torch.ones((T, T), dtype=torch.bool, device=token_ids.device).triu(1)
for block in self.blocks:
    x = block(x, attn_mask=causal_mask, key_padding_mask=None)
```

上述 self.token_embedding、self.position_embedding 和 self.blocks 均在构造函数中建立；堆叠后接最终 LayerNorm 和词表输出层。每层收到同一个 causal_mask，当前样本不含 padding。ViT 不传 causal mask，GPT 不能沿用它的双向设置。

#### 第二步：在 test_language_model.py 编写 GPT 形状、梯度与参考检查

继续向已有 `tests/test_language_model.py` 添加本日检查。

修改 `tests/test_language_model.py`，顶层补充 `from torch import nn` 和 GPTBlock 的导入，新增 `test_gptblock_reference`，将以下四项写入该函数；模型形状 / 梯度另写入 `test_gpt_forward_backward`，未来隔离另写入 `test_gpt_causal_mask`：

1. 取 C=32、heads=4、FFN=128、dropout=0、CPU float64，创建自写 GPTBlock。参考使用 `nn.TransformerEncoderLayer`，设置 batch_first=True、norm_first=True、activation="gelu"、layer_norm_eps=1e-5、bias=True、dropout=0；在关闭所有 dropout 时，这条 pre-LN 计算顺序与作者 Block 对应。
2. 在 torch.no_grad 下，把自写 q_proj / k_proj / v_proj 的 weight 按 dim=0 拼接后复制到 reference.self_attn.in_proj_weight，bias 也按 Q/K/V 顺序拼接；out_proj 单独复制。再把 norm1/norm2 的 gamma/beta 复制到参考 norm1/norm2.weight/bias，两个 FFN Linear 对应 linear1/linear2。
3. 两边输入数值相同的独立 `[B,T,C]` 叶子 Tensor，传相同布尔 causal mask；参考的 Mask 参数名为 src_mask。使用同一固定 probe 反向，比较输出、输入梯度和对应参数梯度，最大绝对差以 1e-6 为目标。
4. 将自写 Block 的 dropout 改为 0.1，检查 Attention 权重、Attention 投影输出、FFN 输出三处 dropout 均已注册并随 train/eval 切换；另外检查 MiniGPT 的 embedding dropout。非零 dropout 不做两套实现的逐值随机输出比较。

#### 第三步：扰动后缀，检查 GPT 前缀 logits 不变

位置 i 只能使用 x 的 0..i；它的 target 是下一 token。训练一次前向能并行计算所有位置，是因为 mask 保证不会提前读取未来输入。

修改后缀时比较前缀 logits；检查应在 eval 且关闭 dropout 下进行。不要因随机 dropout 让同一输入输出不同而误判为因果泄漏。

取 B=2、T=8、k=3，把一份整数输入 clone 后，仅将 `[:,4:]` 改为 `(原ID+1) % V`。两次使用同一 eval 模型，比较 logits 的 `[:,:4]`，最大绝对差以 `1e-6` 为起始容差；取模保证扰动仍是合法 ID。另检查 T=1 能前向、T 超过 context 时明确报错，避免固定长度演示掩盖位置越界。

#### 第四步：先拟合短文本

固定一小段训练文本反复更新，观察 train loss 能否明显下降。若不下降，先查目标错位、mask 方向、optimizer 参数和 step，再扩大文本与训练步数。

在 `tests/test_language_model.py` 新增 `test_gpt_short_text_fit`。函数内直接创建 33 个教学 ID：`torch.tensor([i % 8 for i in range(33)], dtype=torch.long)`，以前 32 个为 x、后 32 个为 y，各加 batch 维得到 `[1,32]`；不读取正式语料文件。新建 vocab_size=8、context_length=32、C=32、heads=4、depth=2、FFN=128、dropout=0 的临时 GPT 和 AdamW，固定这组输入约更新 100 次。记录初始与最终 loss，断言最终更低且有限，再检查 token/position embedding、各 Block、输出 head 的梯度存在且有限；共享的输入 / 输出权重按同一 Parameter 检查。不写正式 history 或 checkpoint。

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

同时将 experiment_dir 改为 `experiments/mini_gpt/week6/runs/gpt`，output_dir 改为 `outputs/mini_gpt`。GPT 创建自己的 GPTBlock 实例，复用 Attention 与 LayerNorm；token embedding、可学习位置嵌入、embedding dropout、最终 LayerNorm 和权重共享由 MiniGPT 管理。

### 运行与自查

```bash
python -m pytest tests/test_language_model.py -q
```

加入 GPTBlock 参考输出/梯度，以及 GPT 的形状、有限梯度、共享权重及未来扰动检查：输入 `[B,T]`，输出 `[B,T,V]`；修改位置 k 之后的输入，前 k+1 个位置的 logits 应在容差内一致。小样本拟合只验证学习路径，正式两组训练在明天运行。

- [ ] logits 只依赖当前与过去的输入，未来隔离通过。
- [ ] 能解释 token embedding、position embedding、无 bias head 的词表维度和输入 / 输出权重共享。
- [ ] 小模型可以前反向和拟合固定短文本。

<a id="day-4"></a>

## Day 4：同口径训练、验证与 checkpoint

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 保存加载（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | state_dict 与重建模型 | 保存 tokenizer 和模型配置关联 |
| [Zero to Hero 作者代码](https://github.com/karpathy/nn-zero-to-hero) | 验证 loss 与训练循环 | 固定比较口径 |

### 操作步骤

#### 第一步：接通训练循环，分开管理训练采样与完整验证

修改已有 `scripts/train_gpt.py`，接通累计 step 循环、定期验证、日志和保存；按需要复用或扩展 `src/petvision/utils/checkpoint.py` 的磁盘读写函数。`src/petvision/training/lm_trainer.py` 继续负责单步更新和完整验证，不能把这些函数另写进模型文件。

训练使用自己的 Generator；验证固定顺序和窗口，不消耗训练采样随机流，否则改变验证频率会改变后续训练批次。

train_step 每次更新前切 train；evaluate_lm 临时切 eval 并关闭梯度，退出时恢复调用前的模式，避免首次验证后 dropout 一直关闭。两组各自从新模型、优化器和 Generator 起步。

#### 第二步：按相同验证窗口训练 Bigram 和 GPT，保存逐步日志

Bigram/GPT 共享 tokenizer JSON、split manifest、验证窗口规则和 token loss 定义。每组独立创建模型、优化器和采样状态，记录参数量和实际训练预算。

```text
训练日志：step、train_loss、val_loss、val_target_tokens
模型记录：config、tokenizer_hash、split_hash、params
```

训练 loss 可以是最近若干批的平均，验证 loss 是固定验证数据的统计；两者记录时写清窗口，不能直接把单个训练批当全部训练集指标。

本轮可约定 train_loss 为最近 eval_interval 次更新所见目标的平均 NLL，按目标 token 数加权；每个 step 只在 optimizer.step 完成后加 1，随后判断是否验证。2000 步、每 200 步一次时，history 的 step 应为 `200,400,...,2000`，共 10 行数据。last 的 step 与最近保存时实际完成更新次数一致，best 按最小 val_loss 选择。

将下面的调度写在 train_gpt.py 的 main 中。train_ids、val_ids、model、optimizer、device、train_generator 已由 Day 2-3 创建，context_length 和 batch_size 从 cfg 读取；循环前设置区间 NLL 和计数为 0：

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

修改 `scripts/train_gpt.py::main()` 的定期验证分支，顶层从 `petvision.utils.checkpoint` 导入 save_checkpoint / load_checkpoint；main() 在 step 循环之前将 best_val_loss 设为正无穷大；定期验证返回 val_metrics 后添加 `if val_metrics["loss"] < best_val_loss` 分支，在其中更新 best_val_loss 并写 best.pt，持平保留更早一次。记录 model_state、config、model_config、vocab_size、context_length、tokenizer_path、tokenizer_hash、split_hash、step、val_loss 和 val_target_tokens；hash 来自实际文件，不能只保存路径后省略核对。main() 在训练循环之前用 yaml.safe_dump(cfg,file,allow_unicode=True,sort_keys=False) 写 experiment_dir/config.yaml；history 由下方 append_lm_history 逐条追加。计时在循环前开始、循环后结束，CUDA 分支前后 synchronize 并统计 peak_allocated_mib；结束后写 run_summary.json，字段含 params、completed_steps、train_seconds、peak_allocated_mib、device、dtype、累计 input_tokens / input_bytes 和 timing_scope。

checkpoint 同时保存模型配置、实际 V、context_length、tokenizer 路径与 hash、split manifest 的 hash，以及本次 step 和验证 loss。加载时核对这些信息，再按结构重建并读取 state_dict。用新建模型重新计算固定验证集，比较保存前后的 loss 与目标 token 数，同时核对数值与计数。

在 `scripts/train_gpt.py` 顶层新增 `append_lm_history(row,path)`：将 path 转为 Path 并创建父目录；打开前检查文件是否不存在或为空，再以 a 模式、encoding="utf-8"、newline="" 打开，用 csv.DictWriter 在新文件 / 空文件写一次表头，每次调用只 writerow(row) 一条；列名改为 step、train_loss、val_loss、val_target_tokens，表头只在新文件 / 空文件写一次。正式分支创建各自 experiment_dir / output_dir，保存 config.yaml，在上面 row 创建后立即调用追加函数；每次仅传这一条 row。新运行使用空日志，已有同名结果先归档；第七周的 resume 才继续现有日志。

每个模型训练结束后读取自己 experiment_dir/history.csv，用 step 为横轴、train_loss / val_loss 为纵轴，保存 `experiment_dir/figures/loss_curves.png`。不要复用要求 epoch / accuracy 的视觉 plot_history；在 train_gpt.py 顶层写 `plot_lm_history(history, experiment_dir)`，函数内用 plt.subplots 创建图，横轴取各行 step，分别 plot 两列 loss 并添加图例；先创建 experiment_dir/figures，savefig 后关闭该 figure。

两组都完成后，在草稿或交互环境读取各自 history，再保存合并对照图 `experiments/mini_gpt/week6/figures/loss_curves.png`。单模型训练入口不硬编码另一模型的结果路径，这样先训练 Bigram 时不会因 GPT 日志尚不存在而失败，第七周 BPE 也可直接复用同一入口。

在 `tests/test_language_model.py` 新增 `test_lm_checkpoint_roundtrip(tmp_path)`，函数内分别创建小 Bigram / MiniGPT、固定输入及验证 IDs，调用 save_checkpoint 保存到 tmp_path，再调用 load_checkpoint 并新建同配置模型加载；同模型同批次 eval logits 保存前后最大差小于 `1e-6`；固定验证窗口的目标计数完全一致、平均 loss 差小于 `1e-6`。正式结果则从 best 对应的验证时点取 val_loss 和计数，不拿 last 的输出与 best 混比。

#### 第四步：读曲线并定位异常

1. 读回两个 history，核对 step 是否为约定验证时点、最后一次是否覆盖训练末步；每行 val_target_tokens 应与该词表下的完整验证长度减一一致。
2. 检查曲线纵轴都是平均 token NLL，横轴都是已完成的更新次数。train_loss 是最近区间的训练统计，val_loss 是完整固定验证窗口统计，先把这两个范围写清。
3. 两者都不下降时，先回到短文本拟合：确认 x/y 错开一位、参数梯度存在、optimizer 绑定当前模型并执行 step；不要先盲目加训练时长。
4. train 持续下降而 val 后期上升时，标出转折区间与 best_step，讨论过拟合；用同一 best 重载验证，排除拿错权重的情况。
5. GPT 未优于 Bigram 时，先核对两组 tokenizer / split hash、验证目标数和预算，再检查 GPT 的因果性。检查通过后保留真实对照，记录可能原因，不用测试集调参，也不要求为了胜出持续重跑。

### 运行与自查

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

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | generate、上下文截断和逐 token 采样 | 实现生成循环 |
| [PyTorch 保存加载（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | 推理 eval 与关闭梯度 | 独立进程加载模型 |

### 操作步骤

#### 第一步：创建 generate.py，从 checkpoint 重建模型与 tokenizer

创建 `scripts/generate.py`，顶层导入 argparse、json、Path、torch、CharTokenizer、BigramLanguageModel 和 MiniGPT。在 main 中解析 checkpoint、prompt、max-new-tokens、seed、output 五个参数，按本步从 checkpoint 重建对象，再调用第二步生成循环和第三步的 JSONL 保存逻辑；文件末尾添加入口保护。只导入脚本不会加载权重或自动生成文本。

参数解析时 max-new-tokens 与 seed 用 type=int，checkpoint 和 output 用 Path，prompt 为字符串；max-new-tokens 要为非负数。

新进程读取结构、context 与 tokenizer，创建模型并加载；固定 seed，在 eval/no_grad 下生成。脚本不应依赖训练程序内存中的对象。

generate.py 的 main 按以下顺序接入：load_checkpoint(args.checkpoint) → 核对 tokenizer 文件 hash → CharTokenizer.load → 核对 tokenizer.vocab_size 与 checkpoint.vocab_size → 根据 model_config.name 新建 Bigram 或 MiniGPT → 严格 load_state_dict → 迁移设备并 eval。MiniGPT 的 heads 仍映射为构造参数 num_heads；context_length 从 checkpoint 读取，不在生成脚本另写 128。main() 用 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` 选择设备并打印，再执行 model.to(device)；本次设备不要求与 checkpoint 记录的训练设备相同。

若后续把 main 中的循环抽成函数，可约定 `generate_ids(model, prompt_ids, context_length, max_new_tokens, device)`，返回完整 token ID 列表；抽取时将循环里的 args 字段改为函数参数，编码与参数解析仍由 main 完成。下方 JSONL 记录还要写实际 device / dtype、prompt 的 OOV 数；第七周再将加载替换为按 tokenizer 类型分派的入口。

重建和加载模型会涉及初始化，因此生成复现用的 seed 在模型加载完成后、采样开始前重新设置。保存前与加载后的两次生成都从相同 prompt 和 seed 开始，比较完整新增 token ID 列表完全一致；以原始 ID 列表核对生成是否一致。

#### 第二步：实现最后位置采样与逐 token 生成循环

生成给定前缀后的下一个 token 时，模型返回 `[B,T,V]`，使用 `logits[:, -1, :]` 得到 `[B,V]`；其他位置对应前缀内部的预测。

先从 checkpoint 还原模型、tokenizer 和 context_length，按第一步说明在加载后重新设置生成 seed。然后把非空 prompt 编码成 `[1,T]` 的 long Tensor；下面片段先放在 generate.py 的 main 中，args 已由 parse_args 得到：

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

每次追加一个 ID 后再次前向，重复 max_new_tokens 次。上下文裁剪影响模型每次可见输入，不要把累积输出文本也截掉。循环后用 `all_ids = token_ids[0].cpu().tolist()` 得到完整 ID 列表，`new_ids = all_ids[len(prompt_ids):]` 得到新增列表，再用 `tokenizer.decode(all_ids)` 得到完整文本；这些实际值用于后面的 JSONL 记录。

先用 3 个已知 prompt ID、max_new_tokens=5 检查最终保留 8 个 ID，前 3 个仍与 prompt 相同；再构造长于 context 的前缀，检查每次送入 forward 的长度不超过 context，累积结果仍持续增长。示例 prompt 必须非空；空 prompt 在本轮明确提示需要输入，避免索引最后位置失败。

<a id="jsonl-output"></a>

#### 第三步：向 samples.jsonl 追加设置、原始 IDs 和生成文本

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

1. 从 samples.jsonl 读取两组同 prompt、seed、生成长度的记录，先核对 checkpoint 和 tokenizer 标识；不要凭终端窗口中最后一段文字判断来源。
2. 保留完整输出，在其中各标一处局部拼写、角色格式或重复片段。没有明显例子就如实写“本次未观察到”，不要另换种子只挑顺畅结果。
3. 分开描述“能看见字符组合规律”和“长句是否连贯”，再结合 Bigram 只看当前 token、GPT 使用前缀解释观察；不要从一个句子直接推断整体模型能力。
4. 把样例观察与验证 loss 放在一起写入 README，注明小语料、有限预算的限制。生成检查还应保留 ID 数量和重载复现结果，不能用文字看起来合理代替这两项检查。

### 运行与自查

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

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度评估说明（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | PPL 定义与 tokenizer 对指标的影响 | 理解下周 BPE 比较边界 |
| [总计划语言模型验收](StudyPlan.md#acceptance) | 同 tokenizer 的基线要求 | 整理本周验证结果 |

### 操作步骤

#### 第一步：读取两组 best 的验证结果，保存对照表并分析样例

在 Python 草稿或交互环境读取 Bigram / MiniGPT 各自的 best checkpoint、history.csv、run_summary.json 和配置，为两组分别建立一个结果字典。字段包含 model、tokenizer_hash、split=val、best_step、val_loss、val_target_tokens、实际完成 steps、params、train_seconds、config、checkpoint；最佳验证值与目标数取自同一个 best 时点。创建 `experiments/mini_gpt/week6/` 后，以 w 模式、UTF-8、newline="" 打开 results.csv，用 csv.DictWriter 写表头和两行真实记录，再读回核对模型名称和来源。不从测试集取值，也不分别挑选不同验证时点的数据。

model、tokenizer_hash、val_loss、steps、params 和时间都对应实际运行。字符两组 tokenizer_hash 必须一致，才有直接比较 token loss 的前提。

若报告 PPL，使用相同口径 loss 的 exp；本周并不要求额外实现复杂 PPL 评估，先保证 NLL 与有效 token 数正确。

例如某个正确目标的预测概率为 0.5，它贡献的 NLL 为 `-ln(0.5)≈0.6931`。先把全部有效目标的 NLL 相加并除以目标数，得到平均 loss，再算 `PPL=exp(loss)`；不是分别计算各窗口 PPL 后取平均。若平均 loss 恰为 ln(2)，PPL 就为 2。不同 tokenizer 的预测单位不同，不能直接把字符 PPL 与 BPE PPL 的大小解释为谁更好。

把 loss 差异、生成例子与架构联系起来：Bigram 只依赖当前 token，GPT 使用因果前缀。也说明小数据、单种子与有限预算的限制。

相同 tokenizer 和验证窗口保证两组指标可直接比较；相同 steps、batch 和 context 约束训练输入量，但两组参数量、计算量和收敛速度仍不同。这是一次基线比较，不能从一次输赢证明某种架构普遍更好。若 GPT 表现较差，先展示因果性、小批过拟合、完整验证和保存加载的检查结果，再结合曲线讨论欠拟合或过拟合；额外实验另存配置与结果，不覆盖首轮记录。

#### 第二步：重写错位窗口、交叉熵展平和生成循环

写 T+1 片段如何得到 x/y、CrossEntropy 的 reshape、逐 token 生成循环。先标 shape，再写操作，不需要再复制整个训练工程。

用草稿实际生成 `[0,1,2]→[1,2,3]` 的错位对，检查展平后的 loss 为有限标量；再让草稿生成循环追加 5 个 token，确认输入只截断上下文、完整输出长度仍增加 5。已有模型测试之外，这三段重写也要被调用核对。

#### 第三步：核对下一周复用的原文、词表、配置与权重

在本周 README 列出三份原文、split manifest、字符词表、两组模型配置和 best/last 的实际路径。读回 manifest 与 checkpoint，核对 split hash、tokenizer hash 和模型结构对应；缺失项写成具体待办。测试原文继续保留；下周变更 tokenizer 时创建新模型并单独报告指标，不直接把两个 token loss 作大小排序。

### 运行与自查

```bash
python -m pytest tests/test_language_model.py -q
```

对照两个 best 指标、history、曲线和 samples，填写 `experiments/mini_gpt/week6/results.csv`。今天以已有验证结果复盘，不为整理表格重新训练，也不评估保留测试原文。复查模型时使用 Day 4 已编写的加载与验证函数核对同一 best。

- [ ] 同 tokenizer 的 GPT/Bigram 对照有真实指标与失败分析。
- [ ] 能解释训练目标、验证统计和自回归生成。
- [ ] 下一周可复用相同原文划分、模型结构与字符基线。

## 本周结束检查

- [ ] 原文划分先于窗口构建，词表仅来自训练数据。
- [ ] Bigram 与 GPT 在相同字符 tokenizer 下完成验证比较，记录真实差异、实现检查和可能原因。
- [ ] 因果信息隔离、保存加载与生成复现检查通过。
- [ ] 配置、tokenizer、曲线、样例和结果口径在本周 README 中齐全。

第七周原样复用原文划分、字符 tokenizer 和两组模型。测试文本继续保留；BPE 使用新词表和从零初始化的新模型。

通过后进入[第七周](week7.md)。
