# 第七周逐日学习指南：简化字节级 BPE、采样与中断续训

> 更新日期：2026-10-24  
> 对应计划：[总学习计划](StudyPlan.md#stage-3)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：手写字节级 BPE 与 Unicode 往返检查，加入 Temperature/Top-k 和完整恢复训练，整理字符/BPE 分开评估的语言模型报告。  
> 环境：本地 Conda `petvision`；复用第六周模型和原文划分。BPE 编解码在 CPU，训练在本地 GPU；续训一致性先用 CPU FP32、dropout=0 的小预算检查。

## 怎么使用这份指南

先确认[上一周指南](week6.md)的基础关卡已通过。第六周 Bigram/GPT 比较、因果性和保存加载应已通过。保留同一份原文 split manifest 与字符 tokenizer，本周为 BPE 新建词表和模型。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/mini_gpt/week7/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── src/petvision/models/llm/tokenizer.py    # Day 1–2：增加 ByteBPETokenizer
├── src/petvision/utils/checkpoint.py       # Day 4：扩展训练状态与随机状态
├── configs/llm/mini_gpt_bpe.yaml           # Day 5：BPE 独立训练配置
├── scripts/
│   ├── train_tokenizer.py                 # Day 2：训练/保存 BPE
│   ├── generate.py                        # Day 3：temperature 与 top-k
│   ├── train_gpt.py                       # Day 4：resume；Day 6：evaluate-only
│   └── compare_tokenizers.py              # Day 5：同原文编码长度和耗时
├── tests/
│   ├── test_tokenizer.py                  # Day 1–2：合并与边界往返
│   ├── test_sampling.py                   # Day 3：过滤、概率与复现
│   └── test_resume.py                     # Day 4：连续/中断结果对照
└── experiments/mini_gpt/week7/
    ├── README.md                         # 已提供
    ├── bpe_tokenizer.json                 # Day 2：字节词表与有序 merge
    ├── tokenizer_comparison.csv           # Day 5：相同原文的编码结果
    ├── samples.jsonl                      # Day 3–5：采样和 BPE 样例
    └── runs/bpe/                          # 配置、history 与曲线
# checkpoints：outputs/mini_gpt_bpe/，模型与 tokenizer 配套保存
```

## 开始前：激活环境并检查已有模块

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

解释器应位于 Conda 的 `petvision` 环境，`petvision.__file__` 应指向当前仓库的 `src/petvision/`。若出现 `No module named petvision`，先确认第二周已建立 `pyproject.toml` 和包目录，再在根目录执行 `python -m pip install -e . --no-deps`；若缺少 torch 或 CUDA 不可用，按[第二周 Day 1 的安装与排错步骤](week2.md#environment)逐项核对。

保留第六周的原文划分、字符 tokenizer 与两组模型，另外建立 BPE 词表和模型。 本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

教学 BPE 从 UTF-8 的 256 个字节 ID 开始，新 merge token 从 256 递增；目标词表 512。先在短字符串手算，再只使用训练原文的前 100,000 个 Unicode 字符学习规则，记录该训练子集的 hash/长度；运行时间允许可用全训练原文，实际配置需明确。每次统计相邻 pair，按最高频选择，平频按字节/token ID 对的字典序决定；从左到右做非重叠替换，encode 按已保存 merge rank 应用。

BPE 模型从零初始化，使用独立 embedding 与输出词表，不把字符 checkpoint 当 BPE 模型直接继续。起始 GPT 结构与第六周相同，context=128、batch=16、2000 steps；token 数相同不代表原文字节预算相同，需分别记录。跨 tokenizer 对照只比较同一原文的编码长度、耗时、参数量与样例，token 级 loss/PPL 分开报告。

## 六天总览

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | 手算 BPE 与字节词表 | 4h | 1h | pair 统计、合并与字节解码 |
| Day 2 | BPE 训练、保存与 Unicode 往返 | 4h | 1h | 可加载的 512 词表与边界检查 |
| Day 3 | Temperature、Top-k 与生成复现 | 4h | 1h | 采样参数与固定 prompt 样例 |
| Day 4 | 完整 checkpoint 与中断续训 | 4h | 1h | 连续/中断训练一致性检查 |
| Day 5 | BPE GPT 小预算训练与编码比较 | 4h | 1h | BPE 训练、编码长度与耗时 |
| Day 6 | 最终语言评估与周总结 | 4h | 1h | 语言模型报告与恢复训练证据 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：手算 BPE 与字节词表

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE) | UTF-8、字节与最小 BPE 合并算法 | 手算相邻 pair 的变化 |
| [minbpe 作者实现](https://github.com/karpathy/minbpe) | BasicTokenizer 与基本 merge 工具 | 理解教学版算法边界 |
| [Hugging Face BPE 课程（官方中文）](https://huggingface.co/learn/llm-course/zh-CN/chapter6/5)、[Python Unicode 指南（官方中文）](https://docs.python.org/zh-cn/3.12/howto/unicode.html) | 最高频 pair 的迭代合并；str.encode 与 bytes.decode | 先用中文讲解理解算法，再核对作者代码 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读字节编码与 BPE |
| 2h | 实现 pair 统计和一次非重叠合并 |
| 1h | 检查平频、重复 pair 与空输入 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 在 `src/petvision/models/llm/tokenizer.py` 增加 ByteBPETokenizer，先把输入字符串编码为 UTF-8 bytes，基础 token 为 0–255；不要先用字符 ID 代替字节 ID。

2. 用 `aaabdaaabac` 等短串手算最高频 pair、替换后序列和新 ID；检查 `aaaa` 中相邻 pair 统计与从左到右非重叠替换的区别。

3. 实现确定性的平频规则与有序 merges；创建 `tests/test_tokenizer.py`，覆盖空串、单字节、重复 pair、无 pair 时提前停止。

4. decode 先拼接所有 token 对应 bytes，再整体解码 UTF-8；一个 token 的 bytes 片段不一定独立构成完整字符。

### 分步操作与自查

#### 第一步：从 abab 手算一次合并

修改第六周已有的 `src/petvision/models/llm/tokenizer.py`，保留 CharTokenizer，在同一文件新增 ByteBPETokenizer 以及 pair 计数、非重叠合并辅助函数。创建 `tests/test_tokenizer.py`，从 `petvision.models.llm.tokenizer` 导入这些对象，将今天的短串、平频和空输入检查写成测试函数。

UTF-8 中 a=97、b=98，初始序列为 `[97,98,97,98]`。相邻 pair 中 `(97,98)` 出现 2 次、`(98,97)` 出现 1 次。

```text
选择 (97,98)，新 token 256 表示 bytes 'ab'
合并后：[256,256]
若再合并，可得到 token 257 表示 bytes 'abab'
```

新 token 保存两个旧 token 的组合及其 bytes，不是将整个文本临时映射为任意编号。

#### 第二步：区分统计与实际替换

`aaaa` 的相邻 aa 统计为 3 次，但一次从左到右非重叠合并会产生两个新 token。替换函数应边匹配边前移两个位置，不能重叠使用同一个字符。

平频时按约定选字典序更小的 pair。用 abc 检查：ab 与 bc 同频，规则应让同样输入在每次训练中产生同样选择。

对 `aaaa`，初始 ID 为 `[97,97,97,97]`；将 `(97,97)` 合并为 256 后，结果应为 `[256,256]`。计数 3 与替换 2 次描述的是不同操作，不能因为计数为 3 就输出三个 256。可分别断言 pair 次数、替换结果和解码结果，避免一种错误被另一种错误掩盖。

#### 第三步：明确三个基础函数

```python
def count_pairs(ids): ...
def merge_pair(ids, pair, new_id): ...
def decode(ids): ...
```

count_pairs 和 merge_pair 可作为 tokenizer.py 内的辅助函数；decode 是 ByteBPETokenizer 的方法，需要使用该对象的词表，不能仅凭任意 ID 列表猜出 bytes。

ByteBPETokenizer 的构造函数先建立 `self.vocab = {i: bytes([i]) for i in range(256)}` 和空的有序 merge 列表；bytes([i]) 表示包含一个值为 i 的字节，不能写成创建 i 个零字节的 bytes(i)。训练每产生新 ID，就把左右 token 的 bytes 拼接后加入 vocab；vocab_size 用只读属性返回 `len(self.vocab)`，调用方读取 `tokenizer.vocab_size`，不加括号，与字符 tokenizer 的属性一致。decode 逐个查 vocab、拼接全部 bytes 后严格解码 UTF-8。

先对短整数列表逐项核对，再将这些函数组合进 tokenizer 类。空串和单 token 没有 pair，应正常返回。

#### 第四步：理解 bytes 与字符的区别

例如 Python 字符串 `'x\t中🙂'` 中 `\t` 表示一个 Tab，整个字符串有 4 个 Unicode 字符、UTF-8 共 9 字节。某个 token 的 bytes 可能只是一个多字节字符的一部分，解码必须先拼完整 bytes。

在 256 个基础字节词表上，`'中'.encode('utf-8')` 为 `[228,184,173]`；检查 decode 这三个 ID 得到「中」。单独的 228 不是完整 UTF-8 字符，严格 decode 应报告错误。再检查空列表解码为空串、单字节 97 解码为 a，明确测试的是 bytes 拼接后的结果。

### 运行与验收

```bash
python -m pytest tests/test_tokenizer.py -q
```

今天先检查字节映射、pair 统计、非重叠替换和确定性规则。`abab` 首次合并应得到 `[256,256]`，`aaaa` 的 aa 计数应为 3；空串和单字节不执行合并。检查只使用很短的教学文本，完整训练在明天运行。

- [ ] 知道字节数与字符数为何可能不同。
- [ ] pair 统计、合并顺序和新 ID 分配可手算核对。
- [ ] 短串训练具有确定性，空输入不会报不必要的错误。

<a id="day-2"></a>
## Day 2：BPE 训练、保存与 Unicode 往返

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [minbpe 作者实现](https://github.com/karpathy/minbpe) | BasicTokenizer 的 train/encode/decode/save/load | 完成独立教学实现 |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE) | 编解码与 Unicode 片段 | 检查完整 bytes 往返 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 复习 merge rank 与序列替换 |
| 2h | 完成 BPE 训练和序列化 |
| 1h | Unicode、空白与加载一致性 |
| 0.5h | 记录词表与训练数据来源 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `scripts/train_tokenizer.py`，只读取第六周 train.txt，按本周配置训练目标词表 512。达到词表预算或没有可合并 pair 时停止，不读取 val/test 来学习规则。

2. 将有序 merge 列表、ID 对应 bytes（用整数列表或 hex 表示）、基础词表、训练数据 hash 与格式版本写入 `bpe_tokenizer.json`。

3. encode 使用训练得到的优先级；decode(encode(text)) 必须对空串、多个空格、Tab、换行、中文、emoji 和混合文本保持原样，不能用 errors='ignore' 掩盖错误。

4. 比较保存前后对相同文本的 ID 与解码结果。检查脚本打印结果即可，无需额外保存检查 CSV。

### 分步操作与自查

<a id="bpe-training-entry"></a>
#### 第一步：固定训练材料和词表预算

创建 `scripts/train_tokenizer.py`，顶层导入 argparse、Path 和 `from petvision.models.llm.tokenizer import ByteBPETokenizer`；训练、保存和打印写在 main，文件末尾加入口保护。继续修改 `src/petvision/models/llm/tokenizer.py` 完成下文的训练与序列化接口，并向 `tests/test_tokenizer.py` 添加保存加载、非法预算与 Unicode 往返测试。

读取第六周 train.txt，只取约定的前 100000 个 Unicode 字符，再转 UTF-8；记录实际字符数、字节数和 hash。256 个基础字节加最多 256 次合并得到目标词表 512。

没有 pair 可合并时提前结束，保存实际词表大小，模型输出维度以这个实际值为准。

把算法接入 train_tokenizer.py：按第二周 argparse 写法接收 `--vocab-size`（int，默认 512）与 `--max-train-chars`（int，默认 100000），后者要求为正。main 读取 `datasets/processed/tinyshakespeare/train.txt` 的 bytes 并按 UTF-8 解码，先按字符截取再转 bytes；新建 ByteBPETokenizer，调用 train(text_subset, vocab_size)，随后 save 到 `experiments/mini_gpt/week7/bpe_tokenizer.json`。父目录先创建，记录子集 hash、字符数和字节数。load 与字符 tokenizer 一样作为类方法，从保存的规则还原对象，不重新训练。

#### 第二步：encode 使用已有规则

对新文本不重新统计频率训练词表；根据已保存 merge rank，从优先级高的规则开始应用。相同文本和相同规则必须得到相同 ID。

序列化需保存 ID、pair 顺序与 bytes 映射。bytes 可转 hex 或整数列表写 JSON，不要求每个 token 都是合法 UTF-8 字符串。

例如规则先合并 `(97,98)→256`，再合并 `(256,256)→257`，新输入 `abab` 应依次变为 `[256,256]`、`[257]`。编码时选择当前相邻 pair 中已学习规则的最小 rank，执行非重叠替换并继续，直到没有可用规则；不根据新输入的频率重新决定合并顺序。

JSON 中 merge 顺序就是规则优先级，加载时保留原顺序。词表使用 hex 时，token 256 的 `ab` 保存为 `6162`；解码先将每个 token 的 hex 还原为 bytes，全部拼接后再 UTF-8 解码。JSON 对象的 key 是字符串，需要在加载时按约定还原 token ID 类型。

给 ByteBPETokenizer 补齐 `train(text, vocab_size)、encode、decode、save、load` 接口。JSON 至少带 `type="byte_bpe"、version=1、base_vocab_size=256、merges、vocab_hex、训练子集hash`；merges 可用 `[left_id,right_id,new_id]` 的有序列表，vocab_hex 可用按 ID 排列的 hex 字符串列表。加载时核对第 r 次合并的新 ID 为 `256+r`、左右 ID 先前已经存在，新 token bytes 等于两侧 bytes 的拼接。

补两个边界：vocab_size 小于 256 时明确报错；空串或单字节文本没有 pair，训练后保留 256 个基础 token，实际合并数为 0。正常训练的 V 为 `256+实际合并数`，不能因目标写 512 就创建不存在的 ID。

#### 第三步：逐条做严格往返检查

```python
cases = ['', ' ', '  a  ', '\t\n', '中文', '🙂', 'hello 中文🙂\n']
for text in cases:
    assert tokenizer.decode(tokenizer.encode(text)) == text
```

再保存、重新加载并对每条文本比较 ID 和解码。不要用 strip、忽略无效字节或归一化空白把检查变成伪通过。

#### 第四步：区分往返检查和模型生成展示

encode 得到的 ID 来自可严格编码为 UTF-8 的原文本，往返必须严格一致；模型任意生成的字节序列可能不完整，展示策略单独记录，不改变 tokenizer 的正确性要求。生成展示可先拼接完整原始 bytes，尝试严格解码；若失败，记录解码失败标记，再用 `raw_bytes.decode("utf-8", errors="replace")` 展示，同时保存原始 token IDs。tokenizer 的标准 decode 仍保持严格模式，不能为了展示而放宽往返检查。

### 运行与验收

```bash
python -m pytest tests/test_tokenizer.py -q
python scripts/train_tokenizer.py --vocab-size 512 --max-train-chars 100000
```

第二条只读取 `datasets/processed/tinyshakespeare/train.txt` 的约定子集，输出 `experiments/mini_gpt/week7/bpe_tokenizer.json`。打印实际字符/字节数、合并次数、V 和耗时，再重新加载该文件核对 Unicode 往返；若实际 V 未达到 512，模型使用实际 V。

- [ ] merge 规则仅来自训练集，保存加载后顺序与编码一致。
- [ ] 空白和 Unicode 边界往返全部通过。
- [ ] 能解释为什么 bytes 拼接后再解码。

<a id="day-3"></a>
## Day 3：Temperature、Top-k 与生成复现

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [nanoGPT 作者代码](https://github.com/karpathy/nanoGPT) | 生成与采样相关代码 | 理解 logits 过滤流程 |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE) | 生成 token 与字节解码关系 | 保留实际输出和 token IDs |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读采样公式并手算小 logits |
| 2h | 扩展生成入口与检查 |
| 1h | 固定设置生成并解释差异 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 扩展 `generate.py` 接收 `--temperature` 与可选 `--top-k`；先除以正数 temperature，再保留 top-k logits、其余填 -inf，softmax 后采样。

2. 创建 `tests/test_sampling.py`：temperature<=0 明确报错；合法 top-k 范围为 1..V，top-k=1 等于贪心；过滤后概率和为 1，相同种子可复现。

3. 先在同一个字符 GPT checkpoint 上，以相同 prompt 和 seed 比较 temperature=0.7/top-k=20 与 temperature=1.0/不启用 top-k；两组保存真实文本和配置到 samples.jsonl。

4. 后续 BPE 生成也保留原始 token IDs。任意模型生成的 bytes 可能不是合法 UTF-8，展示时可用替换字符并记录次数；输入文本的编解码往返仍严格检查。

### 分步操作与自查

#### 第一步：用四个 logits 理解采样顺序

原 logits=`[2,1,0,-1]`，temperature=0.5 后为 `[4,2,0,-2]`，top-k=2 后为 `[4,2,-inf,-inf]`，概率约 `[0.8808,0.1192,0,0]`。

temperature=2 时分布更平缓；top-k 只保留当前得分最高的 k 项。先变换 logits，再 softmax，再采样。

#### 第二步：将采样与模型前向分开

修改已有 `scripts/generate.py`，将下方 sample_next_token 和温度 / 过滤辅助函数定义在文件顶层，模型加载与生成调用仍在 main 内。创建 `tests/test_sampling.py`，测试函数声明 monkeypatch 参数（例如 `def test_sampling_seed(monkeypatch):`），在函数内用 `monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))` 临时加入脚本目录，再 `from generate import sample_next_token` 并导入你定义的过滤函数；Path 从 pathlib 导入。沿用入口保护，导入函数不加载模型或生成样例，不永久修改训练入口的 sys.path。

```python
def sample_next_token(last_logits, temperature=1.0, top_k=None):
    # 返回 [B,1] 整数 token ID
    ...
```

在生成脚本参数解析中添加 temperature（float，默认 1.0）与 top-k（int，默认 None），在生成循环中用 `next_id = sample_next_token(logits[:, -1, :], args.temperature, args.top_k)` 替换第六周直接 softmax / multinomial 的两行。生成长度、上下文裁剪和累计输出仍沿用原循环；不能只接收参数却继续使用旧采样路径。

检查 temperature>0，top_k 为 None 或 1..V。top-k=1 时总选最高项，可以作为确定性最小例子。

将「温度变换与过滤」整理成可单独检查的函数，返回过滤后的 logits；sample_next_token 再执行 softmax 与 multinomial。保留 top-k 的索引后，可先用 `torch.full_like(logits, float('-inf'))` 创建屏蔽表，再 scatter 写回被选中的值。用上面的四项例子分别检查过滤 logits 和概率，而不是只检查最后抽到了哪个 ID。

测试覆盖 temperature=0/负数、top_k=0/V+1 的 ValueError，top_k=None 和 V 不删项；最高 logit 唯一时，top_k=1 的 ID 与 argmax 一致。相同 seed 的复现检查在每次采样前重新设 seed，比较一串结果完全相同，不要求两次连续消费同一随机流后仍得到同一结果。

#### 第三步：复现时固定全部条件

固定 checkpoint、tokenizer、prompt、context、生成长度、temperature、top-k、seed 和运行环境。模型切 eval；采样随机性由 seed 控制，不由 dropout 制造。

#### 第四步：比较两组真实样例

按本周约定生成两组，不挑最好的一段替换完整输出。记录重复、连贯性和格式变化；同 seed 在不同采样分布下可能得到不同文本，这本身是实验现象。

两次运行 `scripts/generate.py` 都传入 `--output experiments/mini_gpt/week7/samples.jsonl`，由第六周的追加写入逻辑保存；record 增加实际 temperature 和 top_k 字段。第一组与第二组保留各自完整记录，不覆盖第六周的样例文件。

两组同时改变了 temperature 和 top-k，因此只说明这两种采样组合的观察结果；若之后想单独归因于温度或 top-k，再增加一次只改变其中一项的对照，放到拓展时间。

### 运行与验收

```bash
python -m pytest tests/test_sampling.py -q
python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 0.7 --top-k 20 --output experiments/mini_gpt/week7/samples.jsonl
python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 1.0 --output experiments/mini_gpt/week7/samples.jsonl
```

先核对四个 logits 例子的概率，再生成两条完整样例。`--output` 沿用第六周新增的输出参数，这次保存到 week7，避免混入 week6 基线；每条记录同时保存本次采样设置与原始 token IDs。

- [ ] 温度和 top-k 的计算顺序正确，模型保持 eval 且关闭梯度。
- [ ] 同 checkpoint、prompt、参数与 seed 可复现。
- [ ] 能解释分布更集中/更分散与实际样例的关系。

<a id="day-4"></a>
## Day 4：完整 checkpoint 与中断续训

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 保存与恢复训练（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | general checkpoint、optimizer 和训练进度 | 区分加载推理与恢复训练 |
| [nanoGPT 作者实现](https://github.com/karpathy/nanoGPT) | 训练状态与恢复相关组织 | 补齐自己的保存字段 |
| [pytest 临时目录（官方英文）](https://docs.pytest.org/en/stable/how-to/tmp_path.html) | tmp_path；普通断言与异常检查沿用第二周 | 将 40 步对照的 checkpoint 写入临时路径 |
| [Python random（官方中文）](https://docs.python.org/zh-cn/3.12/library/random.html#random.getstate)、[NumPy 状态（中文译文）](https://numpy.net.cn/doc/stable/reference/random/generated/numpy.random.get_state.html)、[Torch Generator（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.Generator.html) | getstate/setstate、get_state/set_state；只查下表用到的状态操作 | 保存已经推进的随机流，而非仅保存 seed |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 梳理需要恢复的状态 |
| 2h | 实现 resume 和最小检查 |
| 1h | 比较连续运行与中断运行 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 扩展 checkpoint：model、optimizer、step、config、tokenizer/hash、原文划分 hash、Python/NumPy/Torch/CUDA 随机状态、训练采样 Generator；有 scheduler/GradScaler 时一并保存。同时保存 best_val_loss、best_step、累计处理 token/bytes 数；若在验证间隔中间保存，还需保存当前训练 NLL 累加值与目标计数，保证恢复后日志统计口径连续。字段使用普通容器、Python 数值和 Tensor。

2. 扩展 `train_gpt.py --resume`，恢复后从已完成 step 的下一步继续，训练/验证模式正确。总 steps 表示最终累计目标，不额外再跑一整个预算。

3. 创建 `tests/test_resume.py`：固定小模型 C=32、context=16、batch=4、CPU FP32、dropout=0；比较连续 40 步与 20 步保存后新建对象恢复再到 40 步。检查下一批采样位置、参数、optimizer step/矩状态和最终 loss，起始容差 1e-6。

4. 验证不消耗训练采样 Generator；checkpoint 已恢复后再生成训练 batch。旧的仅含模型权重 checkpoint 继续用于推理，完整续训检查用新增状态格式重新产生的 last。

今天的 40 步与 20+20 步检查在 `test_resume.py` 内固定小模型配置与教学 token 数据，checkpoint 写入 pytest 的临时目录；不读取尚未创建的 `mini_gpt_bpe.yaml`，也不依赖 `outputs/mini_gpt_bpe/last.pt`。正式 BPE 运行与按需恢复安排在 Day 5。

### 分步操作与自查

#### 第一步：明确保存发生在哪个时刻

修改已有 `src/petvision/utils/checkpoint.py`，增加本日的完整训练状态保存 / 恢复功能，保留视觉模型和旧推理 checkpoint 的读取用途。修改 `scripts/train_gpt.py`，在 argparse 中新增 `--resume`（Path，默认 None），main 读配置后按是否提供该参数选择从零创建或恢复，再进入统一训练循环。

选择完成第 K 次 optimizer.step 后保存，记录 step=K。恢复后第一次更新应是 K+1，不能重复 K，也不能从 K+2 开始。

```text
连续：初始化 → 更新 1..40
中断：初始化 → 更新 1..20 → 保存
恢复：新建对象 → 加载状态 → 更新 21..40
```

#### 第二步：分清状态的用途

模型参数决定当前预测；optimizer 的动量/矩估计决定下一步如何更新；scheduler 决定学习率；随机状态决定下一批采样和 dropout。tokenizer 与 split 标识决定 ID 和数据含义。

在第六周推理 checkpoint 的基础上增加这些字段：

| 字段 | 要保存和恢复的内容 |
| --- | --- |
| model_state / model_config | 参数、buffer、完整模型结构和词表大小 |
| optimizer_state | AdamW 的 step、动量和二阶矩等状态 |
| step / config | 已完成更新次数、累计目标与实际训练设置 |
| best_val_loss / best_step / 日志累计值 | 历史最佳验证结果、已处理 token/bytes 数，以及未完成统计区间的 NLL 与目标数 |
| tokenizer / split 标识 | 配套文件路径或内容、hash、原文边界 |
| python_rng / numpy_rng / torch_rng | 各随机数生成器的状态 |
| cuda_rng / train_generator_state | CUDA 随机状态和独立训练采样 Generator 状态 |
| scheduler / scaler | 启用时保存；当前 FP32 小检查可没有这两项 |

优化器也有 state_dict。保存时把 `optimizer.state_dict()` 放入 checkpoint["optimizer_state"]，再用已经学过的 torch.save 写盘。恢复分支先读取 checkpoint，并按保存的结构 / 配置新建模型，迁移到实际训练设备，再创建绑定新模型参数的同类型 AdamW。若启用 scheduler，先构造它并恢复其状态，再加载 optimizer_state；本轮 FP32 小检查不使用 scheduler / scaler。模型和优化器的加载操作为：

```python
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
completed_steps = checkpoint["step"]
```

模型、optimizer 及启用时的 scheduler / scaler 都加载后，再按下表恢复全部随机状态，最后才取得下一批。累计目标取当前核验后的 training.steps，更新循环执行 `range(completed_steps + 1, target_steps + 1)`；旧 checkpoint 缺少 optimizer_state 等必需字段时明确拒绝续训，不能默认为从零开始。

采用与第二周一致的字典/Tensor 格式；NumPy 随机状态里的 ndarray 先转换成普通列表或 Tensor，恢复时再还原为所用 NumPy RNG 的状态格式，使本项目 checkpoint 可以按 `weights_only=True` 加载，相关格式见 [PyTorch 序列化说明](https://docs.pytorch.org/docs/stable/notes/serialization.html)。恢复不仅要重新设置 seed，还要回到保存时已经推进过的具体随机状态。

只有模型参数相同，下一步训练仍可能不同。恢复流程应先构造模型/优化器并加载，再恢复随机状态，最后取下一批数据。

恢复历史 best_val_loss 后再判断新结果是否改善，不能重置为无穷大后覆盖更好的 best；沿用同一输出目录并保留原 best 文件。history 只追加恢复点之后的新验证行，若日志已有比 checkpoint.step 更晚的记录，先归档冲突记录，不能混入本轮。

把随机状态字段对应到真实 API：

| 随机流 | 保存 | 恢复 |
| --- | --- | --- |
| Python random | `random.getstate()` | `random.setstate(state)` |
| NumPy 全局 RNG | `np.random.get_state()` | 还原数组格式后 `np.random.set_state(state)` |
| 独立 NumPy default_rng（若使用） | `rng.bit_generator.state` | 赋回同类型 Generator 的 bit_generator.state |
| Torch CPU | `torch.get_rng_state()` | `torch.set_rng_state(state)` |
| Torch CUDA（可用时） | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(states)` |
| 独立训练采样 Generator | `train_generator.get_state()` | `train_generator.set_state(state)` |

NumPy 全局状态的第二项是 uint32 数组，保存时 `.tolist()`，加载时用 `np.asarray(values, dtype=np.uint32)` 还原，再拼回原五项 tuple。独立 default_rng 不受 `np.random.set_state` 控制，实际使用哪条流就单独保存哪条流；Torch 与采样 Generator 的状态保持 CPU ByteTensor。读取使用 `map_location="cpu", weights_only=True`，模型/优化器加载并迁移后，再恢复各随机流。

#### 第三步：写最小连续/中断对照

创建 `tests/test_resume.py`，从 petvision 包导入 MiniGPT、sample_batch 和实际保存 / 恢复函数。测试内部固定 C=32、context=16、batch=4、CPU FP32、dropout=0 的小配置与教学 token 序列，使用 pytest 的 tmp_path 写 checkpoint。它不读取 Day 5 才创建的 mini_gpt_bpe.yaml，也不依赖正式 BPE 输出文件。

使用 CPU FP32、dropout=0 的小模型，保留相同初始状态和数据。两条运行到 40 步后比较全部模型参数、optimizer step/状态、最后 loss，以及恢复后第一批的采样起点。

只比较最后 loss 很接近还不够，错误的批次或 optimizer 状态也可能偶然给出相近 loss。

对照程序按以下顺序组织：

1. 保存一份共同初始权重；连续路径和中断路径各自新建模型、AdamW 与 Generator，加载这份初值，使用同一教学 token 数据。
2. 连续路径运行 1..40，记录第 21 步实际采样起点；中断路径运行 1..20 后保存完整 last。两条路径使用同一验证时点，验证前后 Generator 状态不变。
3. 再新建模型、优化器与 Generator，加载第 20 步状态；恢复后第一次实际采样应与连续路径第 21 步一致。不要为了查看下一批而额外调用一次采样。
4. 两条路径完成 40 步后，在同一固定 batch、eval 下比较 loss；逐参数求最大绝对差。optimizer 按对应参数检查 step、exp_avg、exp_avg_sq，整数进度完全相同，浮点差小于 `1e-6`。
5. 随机状态另做短往返：保存后分别抽取 Python/NumPy/Torch 数值，恢复后重新抽取，应得到同一串；这项独立检查完成后重置测试状态，再运行训练对照。CPU 小检查不要求 CUDA 才能执行。

#### 第四步：解释总步数语义

steps 是累计目标。last 已完成 500 步，目标为 1000 时再更新 500 步；若目标也是 500，应直接结束。验证不能消费训练采样随机流，恢复后先确认 step 与下一批位置。

### 运行与验收

```bash
python -m pytest tests/test_resume.py -q
```

检查应自行创建小模型和教学 token 数据，并在 pytest 临时目录保存状态。输出或在断言失败时显示下一批采样起点、最大参数差、optimizer 状态差和最终 loss 差；目标为恢复路径与连续路径一致，起始容差 `1e-6`。今天不运行尚未创建的正式 BPE 配置。

- [ ] 模型、优化器、进度、随机状态和 tokenizer 均得到恢复。
- [ ] 连续与中断小规模结果对照通过，未发生重复/跳过更新。
- [ ] 能解释为什么只加载模型参数不足以等价续训。

<a id="day-5"></a>
## Day 5：BPE GPT 小预算训练与编码比较

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度与 tokenizer（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | tokenization 对 PPL 的影响 | 分开报告 token 级指标 |
| [第六周指南](week6.md) | 原文边界、训练和验证口径 | 复用相同原文 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 0.5h | 固定 BPE 模型配置与预算 |
| 2h | 从零训练 BPE GPT 并记录结果 |
| 1h | 同原文编码/耗时对照 |
| 0.5h | 保存生成与配置记录 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `configs/llm/mini_gpt_bpe.yaml`，绑定 bpe_tokenizer.json 和第六周原文划分，从零初始化 V 匹配的新模型；运行 2000 步起始预算。

2. 独立报告 BPE 的 val loss、参数量、训练耗时与处理 token/原文字节数。字符与 BPE 的 context 都为 128 token，但对应原文跨度不同，需在结论中说明。

3. 创建 `scripts/compare_tokenizers.py`，对相同 val 原文计算字符/BPE token 数、UTF-8 字节数与编码耗时；先预热、重复少量次取平均，写入 tokenizer_comparison.csv，注明 OOV 和原文 hash。

4. 用相同可编码 prompt 比较字符/BPE 的生成样例，记录采样参数、token 数和耗时；时间对照说明测量的是编码还是生成，不把不同 tokenizer 的 token/s 直接当原文吞吐率。

### 分步操作与自查

#### 第一步：新 tokenizer 对应新模型词表

读取 bpe_tokenizer.json 的实际 V，重新创建 embedding 和输出 head，再从零训练整个小 GPT。字符 ID 与 BPE ID 含义不同，不能直接把字符模型的同号行解释为同一内容。

保留第六周的原文 train/val/test 边界；分别编码，不因 token 长度变化重新切一份原文。

创建 `configs/llm/mini_gpt_bpe.yaml`，完整起始配置如下；词表大小从实际 BPE 文件读取：

```yaml
model:
  name: mini_gpt
  context_length: 128
  embed_dim: 128
  depth: 2
  heads: 4
  ffn_hidden_dim: 512
  dropout: 0.1
data:
  name: tinyshakespeare
  root: datasets/processed/tinyshakespeare
  split_path: experiments/mini_gpt/week6/split_manifest.json
  tokenizer_path: experiments/mini_gpt/week7/bpe_tokenizer.json
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
experiment_dir: experiments/mini_gpt/week7/runs/bpe
output_dir: outputs/mini_gpt_bpe
```

<a id="tokenizer-dispatch"></a>
在 tokenizer.py 增加供训练和生成共用的 `load_tokenizer(path)`：先读 JSON 的 type；为 char 调用 CharTokenizer.load，为 byte_bpe 调用 ByteBPETokenizer.load，未知类型明确报错。两类对象都提供 encode、decode、save、load 和实际 vocab_size。

将 train_gpt.py、generate.py 中第六周写死的 CharTokenizer.load 改为调用 load_tokenizer，再核对文件 hash；compare_tokenizers.py 也用同一入口分别读两个文件。不能只修改训练脚本，却让生成脚本仍把 BPE 文件当字符表读取。模型、优化器和 embedding/head 均新建，第六周的字符权重保留不动。

<a id="bpe-resume-order"></a>
正式 BPE 训练按以下顺序执行，具体命令见本节下方：

1. 确认 Day 2 的 bpe_tokenizer.json 已保存且能加载，再于今天创建完整的 mini_gpt_bpe.yaml；累计目标写入 `training.steps`，起始为 2000。
2. 首次从零运行，不传 `--resume`；last 按已完成更新步数保存完整训练状态。
3. 只有该次运行确实中断、已产生可读 last，且 `0 < last.step < training.steps` 时，才用同一配置和该文件恢复。例如已完成 500、目标 2000，则继续 1500 步。
4. 若已完成目标步数，直接评估和生成；恢复入口读取后也应退出，不额外增加步数。Day 4 的小规模检查已经验证恢复逻辑，正常跑完 BPE 不需要再人为中断一次。

恢复前核对 tokenizer、split、模型结构与训练设置；以原正式配置的累计目标继续，不能通过改词表或改模型结构套用旧 last。验证及最终测试沿用[第六周窗口规则](week6.md#eval-windows)，各自完整统计 N-1 个目标。

#### 第二步：相同 token 预算不等于相同原文预算

context=128 在字符模型中约为 128 个已编码字符，在 BPE 中可能覆盖更多原文。报告实际 token 数与 UTF-8 字节数，解释比较范围。

累计训练输入 token 数按每批 `x.numel()` 统计；BPE 的处理 bytes 数按输入 ID 对应的 vocab bytes 长度求和，重复采样重复计数，不包括只作为目标而未进入 x 的额外 token。静态训练文件大小、累计处理量和生成输出大小分列，避免把整个语料长度当作本次模型实际看过的字节数。

#### 第三步：编码对照只改变 tokenizer

创建 `scripts/compare_tokenizers.py`，顶层导入 Path、csv、hashlib、time，以及 `from petvision.models.llm.tokenizer import load_tokenizer`。main 读取第六周同一份 val.txt 和两份已保存的 tokenizer；文件末尾加入口保护，以下编码、计时和 CSV 保存都由该入口执行。

```text
固定同一 val 原文 → 字符 encode → token_count / encode_time
固定同一 val 原文 → BPE encode  → token_count / encode_time
```

先预热再计时，重复少量次取平均；记录原文 hash、字节数和 OOV。不能对不同长度的输入计时后直接比较快慢。

两种 tokenizer 先加载完成，固定同一份 val 字符串，各预热一次，再分别用 `time.perf_counter()` 测 3 次 encode、取均值。CSV 每行保存 tokenizer_hash、原文hash、chars、utf8_bytes、token_count、oov_count、mean_encode_seconds、repeats=3。两行的原文 hash/字符数/字节数应一致，token 数允许不同；字符 OOV 如实记录，BPE 对合法 UTF-8 输入不需要 UNK。

将两行实际结果用 csv.DictWriter 保存到 `experiments/mini_gpt/week7/tokenizer_comparison.csv`，创建父目录并写入表头；读取后核对两个 tokenizer 标识、相同原文 hash 和实际行数。这个文件记录编码实验，不能用生成耗时填入 mean_encode_seconds。

#### 第四步：把不同指标分开解释

BPE 验证 loss 单独报告，不能因数字低于字符 loss 就断言更好。比较生成耗时时注明同样 prompt、生成 token 数及实际输出字节数；token/s 依赖 tokenizer。

### 运行与验收

首次运行使用新模型：

```bash
python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml
```

只有该次训练确实中断、且 last 尚未达到累计目标时，才执行下面的恢复命令：

```bash
python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml --resume outputs/mini_gpt_bpe/last.pt
```

训练达到目标后，进行编码对照和生成：

```bash
python scripts/compare_tokenizers.py
python scripts/generate.py --checkpoint outputs/mini_gpt_bpe/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 0.7 --top-k 20 --output experiments/mini_gpt/week7/samples.jsonl
```

编码对照应读取相同 val 原文，保存到 `tokenizer_comparison.csv`；BPE 样例追加到本周 `samples.jsonl`。不要把从零训练和按需恢复两条命令当作必须连续执行的两个训练阶段。

- [ ] BPE 模型与自己的词表配套，没有直接套用字符模型输出头。
- [ ] 跨 tokenizer 的比较使用相同原文，loss/PPL 分开报告。
- [ ] BPE 训练、曲线、编码结果和生成样例均来自实际运行。

<a id="day-6"></a>
## Day 6：最终语言评估与周总结

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度评估说明（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | 相同 tokenizer 下的比较口径 | 整理最终指标 |
| [总计划语言模型验收](StudyPlan.md#acceptance) | BPE、续训和样例要求 | 核查两周成果 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 冻结配置，评估保留测试原文 |
| 1.5h | 整理 tokenizer/采样/续训结论 |
| 1h | 有限参考解释 BPE 与恢复流程 |
| 0.5h | 更新每周 README |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 在配置冻结后，分别评估字符 Bigram、字符 GPT 与 BPE GPT 的 test 原文；为 train_gpt 增加 `--evaluate-only --checkpoint --split`，报告有效目标 token 数、原文范围和各自 tokenizer。

2. 字符两组可比较 token loss；BPE 单列，跨 tokenizer 只解释编码长度、耗时与固定 prompt 样例。将实际结果写入周 README，不按跨词表 loss 数值大小排列优劣。

3. 记录中断续训检查的配置、最大参数差与命令，并有限参考说明 merge rank、Unicode 拼接和应恢复的状态。

4. 完成两周语言模型报告，保留失败例子与尚未通过的关卡。更多上下文长度、更多语料和完整 tokenizer 产品化留作拓展。

### 分步操作与自查

#### 第一步：分别评估各自 tokenizer 的测试原文

继续修改 `scripts/train_gpt.py`，增加下文的 evaluate-only 分支，复用 `src/petvision/training/lm_trainer.py` 中的 evaluate_lm。在 `tests/test_language_model.py` 中补小数据的只评估检查，通过后再执行下面三条正式测试命令。

冻结字符 Bigram、字符 GPT、BPE GPT 的配置和 best，在相同原文 test 边界内分别编码评估。记录有效目标 token 数与窗口规则。

字符两组可直接比较 token loss；BPE 使用独立列或独立表，保留原文范围与词表信息。

`--evaluate-only` 分支在创建训练循环和写训练文件之前返回：按 checkpoint 重建模型与 tokenizer，只读取所选 split，切 eval 并关闭梯度，不更新参数。对长度为 N 的编码序列，用第六周的固定分窗方式统计 `N-1` 个有效目标，保留尾窗并按目标数汇总 NLL。打印模型、tokenizer hash、原文范围、目标数与 loss；测试结果整理到本周总结，不覆盖训练 history 或 checkpoint。

参数解析复用第二周的 argparse：`--evaluate-only` 为 store_true，`--checkpoint` 为 Path，`--split` 限定 val/test；evaluate-only 时必须提供 checkpoint，且与 resume 互斥。先在临时小模型上检查这个分支返回完整目标计数、参数未变，正式运行前后 history 行数与 checkpoint 保持一致。这样下方三条命令确实只评估，不会先训练再评估。

#### 第二步：报告四项证据

报告包含字符基线比较、BPE 往返和编码效率、采样设置与完整样例、续训检查及参数差。每项链接实际配置和结果，不把只有代码文件视为已经实验通过。

#### 第三步：有限参考解释关键实现

用短文本手算一次 merge，说明同频规则；用一组 logits 说明温度/top-k；列出恢复训练状态并解释少一个会怎样。

#### 第四步：整理仍未通过的点

如果 Unicode 往返、续训一致性或指标口径仍不清楚，记录具体失败输入和下一步。更多语料、上下文长度和 tokenizer 产品化放入后续路线。

### 运行与验收

```bash
python scripts/train_gpt.py --config configs/llm/bigram_char.yaml --evaluate-only --checkpoint outputs/bigram_char/best.pt --split test
python scripts/train_gpt.py --config configs/llm/mini_gpt.yaml --evaluate-only --checkpoint outputs/mini_gpt/best.pt --split test
python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml --evaluate-only --checkpoint outputs/mini_gpt_bpe/best.pt --split test
```

字符两组的 tokenizer hash 与有效目标数应相同；BPE 的目标数按自己的编码长度计算，单独报告。三条命令都只评估，不开始训练。将测试结果、采样观察、BPE 往返和 Day 4 续训检查一起写入本周总结。

- [ ] 原文测试集在配置冻结后评估，指标口径明确。
- [ ] 采样、BPE 往返和恢复训练证据齐全。
- [ ] 能解释自回归概率、tokenizer 与模型 checkpoint 的对应关系。

## 本周结束检查

- [ ] 教学 BPE 训练/保存/编解码完成，Unicode 与空白往返通过。
- [ ] Temperature/Top-k 行为正确，固定条件生成可复现。
- [ ] 完整中断续训与连续训练小规模对照通过。
- [ ] 字符/BPE 指标分开，编码长度、耗时、原始输出与失败分析齐全。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

通过后进入[第八周](week8.md)。

