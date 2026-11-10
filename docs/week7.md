# 第七周：简化字节级 BPE、采样与中断续训

手写字节级 BPE，给生成加入采样参数，再检查中断续训。最后分别评估字符模型和 BPE 模型，说明哪些指标可以直接比较。

对应[总计划第 7 周](StudyPlan.md#week-7)。安排为 6 个学习日，每天 4 小时任务和 1 小时机动，共 30 小时；第 7 天休息。Day 1 可从任意星期开始，卡住时先补当前步骤，再顺延后面的任务。

环境：本地 Conda `petvision`；复用第六周模型和原文划分。BPE 编解码在 CPU，训练在本地 GPU；续训一致性先用 CPU FP32、dropout=0 的小预算检查。

## 开始前

字符 Bigram/GPT 已完成验证对照、因果性检查和保存加载。保留第六周原文划分和字符词表。

按当天的阅读范围、操作步骤和检查顺序学习。通用的路径、代码片段和记录约定见[执行说明](README.md#working-rules)；实际进度、命令和结果写在[本周实验记录](../experiments/mini_gpt/week7/README.md)。

更新日期：2026-11-10。

## 本周文件

按注释中的学习日逐步创建或修改文件；已有文件继续使用，结果在运行后保存。

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
    └── runs/bpe/                          # 配置、history、run_summary、评估 JSON 与曲线
# checkpoints：outputs/mini_gpt_bpe/，模型与 tokenizer 配套保存
```

## 环境检查

以下命令从项目根目录执行，每个学习日先激活同一个环境：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, yaml, pytest, petvision; print(sys.executable); print(torch.__version__); print(petvision.__file__)"
python -m pip check
```

确认解释器属于 `petvision`，`petvision.__file__` 指向当前仓库的 `src/petvision/`。导入失败时先在根目录运行 `python -m pip install -e . --no-deps`；依赖或 CUDA 问题按[第二周环境排查](week2.md#environment)处理。

保留第六周的原文划分、字符 tokenizer 与两组模型，另外建立 BPE 词表和模型。本周沿用已有依赖；版本没有变化时复用原环境快照，新增或调整依赖后才更新实际安装记录。当天脚本和检查文件在对应任务中创建，完成实现后再运行下方命令。

## 固定条件与接口约定

<a id="bpe-terms"></a>
本周的 BPE 是 Byte Pair Encoding（字节对编码）。训练 tokenizer 时统计相邻 token，学习合并规则；GPT 参数通过反向传播训练，安排在 Day 5。

| 名称 | 具体含义 |
| --- | --- |
| byte / 字节 ID | 一个字节取值 0..255；初始词表用这个数值作为 ID |
| pair | 当前序列中相邻两个 token ID 组成的有序对，例如 (97,98)；顺序不可交换 |
| merge | 把一个指定 pair 替换成新 ID 的规则；新 ID 对应原来两个 token 的 bytes 拼接 |
| rank | 合并规则的学习顺序，从 0 开始；数值越小，编码时优先级越高 |
| V / vocab_size | 基础字节和新合并 token 的总数；目标 512 包含最初的 256，不是额外加 512 |
| temperature / top_k | 生成时调整分布的温度 / 第 k 高分的过滤阈值；阈值处并列时保留数可能超过 k，不修改 tokenizer 规则或模型权重 |

教学 BPE 从 UTF-8 的 256 个字节 ID 开始，新 merge token 从 256 递增；目标词表 512。先在短字符串手算，再只使用训练原文的前 100,000 个 Unicode 字符学习规则，记录该训练子集的 hash/长度；运行时间允许可用全训练原文，实际配置需明确。算法概念先读教材的子词嵌入；字节级实现按 minbpe 的 BasicTokenizer：每次从左到右统计相邻 pair，按最高频选择，平频保留当前序列中先遇到的 pair；从左到右做非重叠替换，encode 按已保存 merge rank 应用。本轮不加正则预切分或特殊 token；空输入提前停止、JSON 保存与显式严格解码参数是项目补充。

BPE 模型从零初始化，按 BPE 的实际 V 新建 embedding 与输出词表，不把字符 checkpoint 当 BPE 模型直接继续；模型内部仍按第六周共享 token embedding / head 权重。起始 GPT 结构与第六周相同，context=128、batch=16、2000 steps；token 数相同不代表原文字节预算相同，需分别记录。跨 tokenizer 对照只比较同一原文的编码长度、耗时、参数量与样例，token 级 loss/PPL 分开报告。

## 六天安排

| 学习日 | 修改哪些文件、按什么顺序做 | 做完应看到什么 |
| --- | --- | --- |
| [Day 1](#day-1) | 在 `models/llm/tokenizer.py` 增加字节 BPE，创建 tokenizer 测试 | pair 统计与非重叠替换正确；先遇到的同频 pair 优先 |
| [Day 2](#day-2) | 补齐 tokenizer 方法并通过测试，再创建 `scripts/train_tokenizer.py`，保存 `bpe_tokenizer.json` | 只用 train 学 merges；空白、中文、emoji 严格往返；保存加载一致 |
| [Day 3](#day-3) | 扩展 `scripts/generate.py`，创建 `tests/test_sampling.py` | 温度、top-k 阈值和并列项正确；固定 seed 的 IDs 可复现 |
| [Day 4](#day-4) | 扩展 checkpoint/训练入口，创建 `tests/test_resume.py` | CPU 小模型连续 40 步与 20+20 步的状态、下一批和 loss 一致 |
| [Day 5](#day-5) | 创建 BPE GPT 配置和 `scripts/compare_tokenizers.py` | 新模型完成预算；同一 val 原文的编码长度、字节数与时间有记录 |
| [Day 6](#day-6) | 给训练入口增加仅评估模式，更新本周报告 | 三个模型各自完整测试；字符 loss 可比较，BPE 单列 |

<a id="day-1"></a>

## Day 1：手算 BPE 与字节词表

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [D2L 14.6.2 字节对编码（主概念材料）](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html) | 从符号对频率到合并规则 | 手算一次最高频合并；区分教材字符/词尾符号与本周 UTF-8 字节起点 |
| [minbpe BasicTokenizer（本周实现依据）](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py)、[base.py](https://github.com/karpathy/minbpe/blob/master/minbpe/base.py) | train、encode、decode、get_stats 与 merge | 核对同频选择、非重叠替换、merge rank 和 errors="replace" |
| [Python Unicode 指南（官方中文）](https://docs.python.org/zh-cn/3.12/howto/unicode.html) | str.encode 与 bytes.decode | 检查中文字节拼接与两种解码模式 |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE)（按需选看） | UTF-8、字节与最小 BPE 合并算法 | 对照手算中不清楚的步骤 |

### 操作步骤

基础字节 ID 为 0..255。可用 `aaabdaaabac` 手算最高频 pair，用 `aaaa` 区分相邻统计与非重叠替换。

#### 第一步：手算 abab、aaaa 和 caba 的计数与替换

UTF-8 中 a=97、b=98，初始序列为 `[97,98,97,98]`。相邻 pair 中 `(97,98)` 出现 2 次、`(98,97)` 出现 1 次。

```text
选择 (97,98)，新 token 256 表示 bytes 'ab'
合并后：[256,256]
若再合并，可得到 token 257 表示 bytes 'abab'
```

新 token 保存两个旧 token 的组合及其 bytes，不是将整个文本临时映射为任意编号。

`aaaa` 的相邻 aa 统计为 3 次，但一次从左到右非重叠合并会产生两个新 token。替换函数应边匹配边前移两个位置，不能重叠使用同一个字符。

平频时按 minbpe 选择当前序列中先遇到的 pair。用 `caba` 检查：ca、ab、ba 各出现一次，应先合并 ca，得到 `[256,98,97]`；若先合并 ab，就说明误用了字典序规则。

对 `aaaa`，初始 ID 为 `[97,97,97,97]`；将 `(97,97)` 合并为 256 后，结果应为 `[256,256]`。计数 3 与替换 2 次描述的是不同操作，不能因为计数为 3 就输出三个 256。可分别断言 pair 次数、替换结果和解码结果，避免一种错误被另一种错误掩盖。

#### 第二步：在 tokenizer.py 完成计数、合并与字节 decode

修改第六周已有的 `src/petvision/models/llm/tokenizer.py`，保留 CharTokenizer，在同一文件新增 ByteBPETokenizer 以及 pair 计数、非重叠合并辅助函数。

```python
def count_pairs(ids): ...
def merge_pair(ids, pair, new_id): ...


class ByteBPETokenizer:
    def __init__(self): ...
    def decode(self, ids, errors="replace"): ...
```

count_pairs 和 merge_pair 可作为 tokenizer.py 内的辅助函数；decode 是 ByteBPETokenizer 的方法，需要使用该对象的词表，不能仅凭任意 ID 列表猜出 bytes。

count_pairs 返回 `{(left_id, right_id): 次数}` 字典；merge_pair 返回替换后的新列表。没有 pair 时返回空计数字典，训练循环在取最大值前退出。按 minbpe 写为 `max(counts, key=counts.get)`；这行仅在 counts 非空时执行。count_pairs 必须从左到右向普通 dict 插入首次出现的 pair，不能先转成 set 或按 ID 排序。

先独立实现两个辅助函数，不要一开始就把统计和替换揉进训练循环：

1. count_pairs 创建空计数字典；从位置 0 遍历到倒数第二个位置，每次取当前 ID 与下一个 ID 作为 pair，将对应次数加一。位置每次前移一格，因此允许相邻 pair 重叠。
2. 用 `[97,98,97,98]` 检查字典恰有两个键及正确次数，再用 `[97,97,97,97]` 检查同一 pair 计数为 3；空列表和单元素列表返回空字典。
3. merge_pair 新建结果列表，将游标设为 0。若当前位置及下一位置刚好等于指定 pair，就追加 new_id，游标前移两格；否则追加当前位置原 ID，游标只前移一格。
4. 游标到末尾时返回新列表；如果最后只剩一个 ID，按未匹配处理保留下来。不要在原列表上边删除边遍历，避免跳过位置。
5. 依次检查 `aaaa`、`aaa`、没有目标 pair 的序列，结果应分别体现两次替换、一次替换加一个原 ID、原样返回。确认输入列表没有被意外修改，再把这两个函数交给 train 和 encode 调用。

这里 max 比较每个 pair 的次数；出现相同最大值时，Python 保留迭代中先遇到的键。dict 保留插入顺序，因此从左到右统计就得到作者的同频规则。以 caba 为例，键依次为 `(99,97)、(97,98)、(98,97)`，三者同频时选第一项。训练每轮替换后重新统计当前序列；encode 则只遵循已保存规则的 rank，不学习新规则。

ByteBPETokenizer 的构造函数先建立 `self.vocab = {i: bytes([i]) for i in range(256)}` 和空的有序 merge 列表；bytes([i]) 表示包含一个值为 i 的字节，不能写成创建 i 个零字节的 bytes(i)。训练每产生新 ID，就把左右 token 的 bytes 拼接后加入 vocab；vocab_size 用只读属性返回 `len(self.vocab)`，调用方读取 `tokenizer.vocab_size`，不加括号，与字符 tokenizer 的属性一致。decode 逐个查 vocab，先拼接全部 bytes，再调用 UTF-8 decode，并把 errors 参数传入；默认 errors="replace" 与 minbpe 一致。往返检查显式传 errors="strict"，生成展示使用默认策略；未知 ID 仍应报错。CharTokenizer 的 decode 接口无需改变。

先对短整数列表逐项核对，再将这些函数组合进 tokenizer 类。空串和单 token 没有 pair，应正常返回。

#### 第三步：创建 test_tokenizer.py，检查 pair 规则与字节解码

创建 `tests/test_tokenizer.py`，从 `petvision.models.llm.tokenizer` 导入 ByteBPETokenizer、count_pairs、merge_pair。新增 `test_bpe_pair_counts` 核对 abab / aaaa 的相邻计数，`test_bpe_non_overlapping_merge` 核对 aaaa / aaa 的替换结果，`test_bpe_first_seen_tie` 核对 caba 同频时先选 ca，`test_bpe_decode_bytes` 核对完整中文字节拼接及 strict / replace 解码；空串、单 ID 分别作为这些函数的边界输入。

例如 Python 字符串 `'x\t中🙂'` 中 `\t` 表示一个 Tab，整个字符串有 4 个 Unicode 字符、UTF-8 共 9 字节。某个 token 的 bytes 可能只是一个多字节字符的一部分，解码必须先拼完整 bytes。

在 256 个基础字节词表上，`'中'.encode('utf-8')` 为 `[228,184,173]`；检查 decode 这三个 ID 得到「中」。单独的 228 不是完整 UTF-8 字符：`decode([228], errors="strict")` 应抛出 UnicodeDecodeError，默认 `decode([228])` 则得到替换字符 U+FFFD。再检查空列表解码为空串、单字节 97 解码为 a，明确测试的是 bytes 拼接后的结果。

### 运行与自查

```bash
python -m pytest tests/test_tokenizer.py -q
```

今天先检查字节映射、pair 统计、非重叠替换和确定性规则。`abab` 首次合并应得到 `[256,256]`，`aaaa` 的 aa 计数应为 3；空串和单字节不执行合并。检查只使用很短的教学文本，完整训练在明天运行。

- [ ] 知道字节数与字符数为何可能不同。
- [ ] pair 统计、合并顺序和新 ID 分配可手算核对。
- [ ] 短串的 pair 选择与替换具有确定性，空输入不会报不必要的错误。

<a id="day-2"></a>

## Day 2：BPE 训练、保存与 Unicode 往返

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [minbpe 作者实现](https://github.com/karpathy/minbpe) | BasicTokenizer 的 train/encode/decode/save/load | 完成独立教学实现 |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE) | 编解码与 Unicode 片段 | 检查完整 bytes 往返 |

### 操作步骤

#### 第一步：完成 tokenizer.py 的 BPE train 方法

修改 `src/petvision/models/llm/tokenizer.py` 的 ByteBPETokenizer.train / encode / save / load 方法，先完成本步的 train，再在第二步完成 encode / save / load。

在 ByteBPETokenizer.train 内按以下顺序完成训练算法：

1. 检查目标词表大小至少为 256。本轮每次训练都从新的 256 字节词表和空 merges 开始，不沿用上次训练的规则。
2. 将传入字符串编码为 UTF-8 bytes，再转成整数列表；此时每个整数就是一个基础 token ID。
3. 当实际词表还未达到目标时，调用 count_pairs；计数字典为空就停止，否则按 Day 1 的频率与平频规则选一个 pair。
4. 将当前词表大小作为 new_id，拼接左右 token 已保存的 bytes 得到新词条，并把 `[left_id,right_id,new_id]` 追加到有序 merges。
5. 调用 merge_pair 更新当前 ID 列表，然后回到第三步，对更新后的序列重新统计；不要重复使用第一轮的频率。
6. 用 `abab`、目标 V=258 手算并运行，确认依次形成 256 和 257；用空串确认能提前停止。短例子通过后，才由 train_tokenizer.py 传入正式训练子集并保存文件。

#### 第二步：按 merge rank 实现 encode，并保存加载规则

对新文本不重新统计频率训练词表；根据已保存 merge rank，从优先级高的规则开始应用。相同文本和相同规则必须得到相同 ID。

序列化需保存 ID、pair 顺序与 bytes 映射。bytes 可转 hex 或整数列表写 JSON，不要求每个 token 都是合法 UTF-8 字符串。

例如规则先合并 `(97,98)→256`，再合并 `(256,256)→257`，新输入 `abab` 应依次变为 `[256,256]`、`[257]`。编码时选择当前相邻 pair 中已学习规则的最小 rank，执行非重叠替换并继续，直到没有可用规则；不根据新输入的频率重新决定合并顺序。

实现 encode 时先从 merges 建立“pair → rank / new_id”的查询表。把新文本转成基础字节 ID 后，扫描当前相邻 pair，只保留查询表中已有的候选；候选为空就返回当前序列，否则选择 rank 最小的一对，调用 merge_pair，再重新扫描。整个过程只读词表与规则，不新增 ID。用一个新文本里高频 pair 与低 rank pair 不一致的短例子，确认选择依据是已保存 rank，而不是当前频率。

JSON 中 merge 顺序就是规则优先级，加载时保留原顺序。词表使用 hex 时，token 256 的 `ab` 保存为 `6162`；解码先将每个 token 的 hex 还原为 bytes，全部拼接后再 UTF-8 解码。JSON 对象的 key 是字符串，需要在加载时按约定还原 token ID 类型。

给 ByteBPETokenizer 补齐 `train(text, vocab_size)、encode、decode、save、load` 接口。JSON 至少带 `type="byte_bpe"、version=1、base_vocab_size=256、merges、vocab_hex、训练子集hash`；merges 可用 `[left_id,right_id,new_id]` 的有序列表，vocab_hex 可用按 ID 排列的 hex 字符串列表。加载时核对第 r 次合并的新 ID 为 `256+r`、左右 ID 先前已经存在，新 token bytes 等于两侧 bytes 的拼接。

补两个边界：vocab_size 小于 256 时明确报错；空串或单字节文本没有 pair，训练后保留 256 个基础 token，实际合并数为 0。正常训练的 V 为 `256+实际合并数`，不能因目标写 512 就创建不存在的 ID。

#### 第三步：在 test_tokenizer.py 检查训练、往返与保存加载

在已有 `tests/test_tokenizer.py` 中新增 `test_bpe_roundtrip`，用本节 cases 逐条 strict 解码；新增 `test_bpe_save_load(tmp_path)`，比较保存前后 merges、vocab 和相同文本的 IDs；新增 `test_bpe_vocab_budget`，检查预算小于 256 报错、空文本提前停止；新增 `test_bpe_encode_merge_rank`，检查新输入按保存的 rank 合并。

```python
cases = ['', ' ', '  a  ', '\t\n', '中文', '🙂', 'hello 中文🙂\n']
for text in cases:
    assert tokenizer.decode(tokenizer.encode(text), errors="strict") == text
```

将上面的 cases 放进 test_bpe_roundtrip，逐条检查。再在 test_bpe_save_load 中保存、重新加载，对每条文本比较 ID 列表与 strict 解码结果；空格、Tab 和换行逐字符相同。不要用 strip、忽略无效字节或归一化空白绕过检查。

encode 得到的 ID 来自可严格编码为 UTF-8 的原文本，往返用 `decode(ids, errors="strict")` 检查，必须保持原文；默认 decode 的替换策略用于模型生成，与 minbpe 一致。生成时先保留完整 IDs 与拼接后的 raw_bytes，再尝试 strict 解码以记录 decode_ok；失败时用默认 decode 展示替换字符，同时保存 raw_bytes_hex。给同一组无效字节分别检查 strict 报错与默认替换，不能用 errors="ignore" 或 strip 掩盖问题。

<a id="bpe-training-entry"></a>

#### 第四步：完成 train_tokenizer.py，再训练正式子集

创建 `scripts/train_tokenizer.py`，顶层导入 argparse、Path 和 `from petvision.models.llm.tokenizer import ByteBPETokenizer`；训练、保存和打印写在 main，文件末尾加入口保护。

读取第六周 train.txt，只取约定的前 100000 个 Unicode 字符，再转 UTF-8；记录实际字符数、字节数和 hash。256 个基础字节加最多 256 次合并得到目标词表 512。

没有 pair 可合并时提前结束，保存实际词表大小，模型输出维度以这个实际值为准。

在 `scripts/train_tokenizer.py` 定义 parse_args(argv=None)，用 argparse.ArgumentParser 添加 `--vocab-size`（type=int，默认 512）与 `--max-train-chars`（type=int，默认 100000）；main() 调用 parse_args，先检查后者为正，再读取训练文本。main 读取 `datasets/processed/tinyshakespeare/train.txt` 的 bytes 并按 UTF-8 解码，先按字符截取再转 bytes；新建 ByteBPETokenizer，调用 train(text_subset, vocab_size)，随后 save 到 `experiments/mini_gpt/week7/bpe_tokenizer.json`。父目录先创建，记录子集 hash、字符数和字节数。重新加载时调用第二步已实现的类方法 load，从保存规则还原对象，不重新训练。

读取 train.txt 后先与第六周 split_manifest 的 train hash 核对，再计算截取子集的 hash；两者分别保存为原训练 split 标识和 BPE 学习子集标识。已有 bpe_tokenizer.json 时先核对这些条件，改变词表或子集应另存版本，不能覆盖仍被 checkpoint 引用的 tokenizer。

### 运行与自查

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

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) | GPT.generate 中 temperature、top-k 阈值和 multinomial | 按第 k 高分过滤，保留阈值处的并列项 |
| [Tokenizer 作者课程](https://www.youtube.com/watch?v=zduSFxRajkE) | 生成 token 与字节解码关系 | 保留实际输出和 token IDs |

### 操作步骤

先以同一个字符 GPT 和固定 prompt/seed 比较两组采样参数；BPE 生成在 Day 5 再接入，届时保留原始 token IDs 和解码状态。

#### 第一步：手算温度缩放、top-k 过滤和 softmax 概率

原 logits=`[2,1,0,-1]`，temperature=0.5 后为 `[4,2,0,-2]`，top-k=2 后为 `[4,2,-inf,-inf]`，概率约 `[0.8808,0.1192,0,0]`。

temperature=2 时分布更平缓；top-k 按第 k 高分设阈值，保留大于等于阈值的项。无并列时正好保留 k 项；有并列时可能更多。先变换 logits，再 softmax，再采样。

temperature 是正数缩放参数，先计算 `scaled_logits=logits/temperature`：小于 1 放大分数差距，大于 1 缩小差距，等于 1 保持原分数。本例 top_k=2 的阈值没有并列，因此保留两个候选；k 控制候选过滤，不是生成 token 的数量。其余分数设为 -inf 后 softmax 概率为 0。最后按概率随机抽取一个 ID，例如用 `torch.multinomial(probabilities, num_samples=1)`；它不等同于每次都取最大概率的 argmax。温度不要与第六周表示序列长度的 T 混用。

#### 第二步：在 generate.py 完成采样函数与参数接入

修改已有 `scripts/generate.py`，在文件顶层定义下方 `sample_next_token(last_logits,temperature=1.0,top_k=None)` 和 `filter_logits(last_logits,temperature=1.0,top_k=None)`，模型加载与生成调用仍在 main 内。

```python
def sample_next_token(last_logits, temperature=1.0, top_k=None):
    # 返回 [B,1] 整数 token ID
    ...
```

在生成脚本参数解析中添加 temperature（float，默认 1.0）与 top-k（int，默认 None），在生成循环中用 `next_id = sample_next_token(logits[:, -1, :], args.temperature, args.top_k)` 替换第六周直接 softmax / multinomial 的两行。生成长度、上下文裁剪和累计输出仍沿用原循环；不能只接收参数却继续使用旧采样路径。

在脚本顶层导入 math，检查 `math.isfinite(temperature) and temperature > 0`，top_k 为 None 或正整数；非 None 时令 effective_k=min(top_k,V)，与作者实现一致。top-k=1 且最高分唯一时，总选这一项，可以作为确定性最小例子。

作者只使用 torch.topk 返回的第 k 高数值，不使用并列项的索引决定去留。以 `[2,1,1,0]`、top_k=2 为例，阈值为 1，输出应为 `[2,1,1,-inf]`，实际保留三项；top_k=1 遇到并列最高分也可能保留多项。因此只在最高分唯一时检查它与 argmax 一致。

在 `scripts/generate.py::filter_logits` 中完成温度校验、除法与 top-k 阈值过滤，返回 `[B,V]` logits；`sample_next_token` 调用 filter_logits 后再执行 softmax 与 multinomial。沿最后一维取 effective_k 个最大值，只取其中最后一个值作为每行阈值，形状保留 `[B,1]`；用 masked_fill 将严格小于阈值的项填为 -inf。这里用 `<`，不能写 `<=`，也不按 topk 索引 scatter 成恰好 k 项。用上面的四项例子分别检查过滤 logits 和概率，而不是只检查最后抽到了哪个 ID。

按下面顺序写函数体，先用四项手算例子调通，再接回生成循环：

1. 确认输入是最后位置的 `[B,V]` logits，验证 temperature 和 top_k 参数；不要把 `[B,T,V]` 的整段输出直接交给这里。
2. 用 temperature 除每个 logit，得到缩放后的新张量，保留原始输入便于比较。
3. top_k 为 None 时跳过过滤；否则先取 effective_k=min(top_k,V)，用 topk 的第 effective_k 高数值作阈值，仅屏蔽严格小于阈值的项。top_k=V 或大于 V 时保留全部项。
4. 返回过滤后的 logits，用 `[2,1,0,-1]`、temperature=0.5、top_k=2 检查结果恰为 `[4,2,-inf,-inf]`。
5. sample_next_token 对过滤结果沿词表轴做 softmax，检查每行概率和约为 1，再每行抽取一个 ID，输出形状保持 `[B,1]`。
6. 用 B=1 和 B=2 各调用一次，确认两行各自过滤和采样；最后接回 generate.py，每次生成只追加这一列 ID。

#### 第三步：编写 test_sampling.py 的参数、过滤和复现检查

创建 `tests/test_sampling.py`，测试函数声明 monkeypatch 参数（例如 `def test_sampling_seed(monkeypatch):`），在函数内用 `monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))` 临时加入脚本目录，再 `from generate import sample_next_token` 并导入 filter_logits；Path 从 pathlib 导入。沿用入口保护，导入函数不加载模型或生成样例，不永久修改训练入口的 sys.path。

`tests/test_sampling.py` 中分别用 `test_sampling_invalid_parameters` 检查参数报错，`test_sampling_topk_threshold` 检查四项 logits 及并列阈值，`test_sampling_seed` 检查固定种子生成的 ID 序列相同。

测试覆盖 temperature=0/负数/NaN/无穷大、top_k=0/负数/非整数的 ValueError；top_k=None、V、V+1 都不删项，并列阈值例子保留三项；最高 logit 唯一时，top_k=1 的 ID 与 argmax 一致。相同 seed 的复现检查在每次采样前重新设 seed，比较一串结果完全相同，不要求两次连续消费同一随机流后仍得到同一结果。

#### 第四步：固定参数生成两组样例，再核对相同设置的 IDs

先运行下方两条生成命令，将 checkpoint、tokenizer、prompt、context、生成长度、temperature、top-k、seed 和环境写入对应记录。然后选定其中一组，在新进程用完全相同参数再运行，读回两条记录并比较新增 token ID 列表，要求逐项相同。模型保持 eval；采样随机性由 seed 控制。

按本周约定生成两组，不挑最好的一段替换完整输出。记录重复、连贯性和格式变化；同 seed 在不同采样分布下可能得到不同文本，这本身是实验现象。

两次运行 `scripts/generate.py` 都传入 `--output experiments/mini_gpt/week7/samples.jsonl`，由第六周的追加写入逻辑保存；record 增加实际 temperature 和 top_k 字段。第一组与第二组保留各自完整记录，不覆盖第六周的样例文件。

两组同时改变了 temperature 和 top-k，因此只说明这两种采样组合的观察结果；若之后想单独归因于温度或 top-k，再增加一次只改变其中一项的对照，放到拓展时间。

### 运行与自查

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

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [PyTorch 保存与恢复训练（中文译文）](https://docs.pytorch.ac.cn/tutorials/beginner/saving_loading_models.html) | general checkpoint、optimizer 和训练进度 | 区分加载推理与恢复训练 |
| [nanoGPT 作者实现](https://github.com/karpathy/nanoGPT) | 训练状态与恢复相关组织 | 补齐自己的保存字段 |
| [pytest 临时目录（官方英文）](https://docs.pytest.org/en/stable/how-to/tmp_path.html) | tmp_path；普通断言与异常检查沿用第二周 | 将 40 步对照的 checkpoint 写入临时路径 |
| [Python random（官方中文）](https://docs.python.org/zh-cn/3.12/library/random.html#random.getstate)、[NumPy 状态（中文译文）](https://numpy.net.cn/doc/stable/reference/random/generated/numpy.random.get_state.html)、[Torch Generator（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.Generator.html) | getstate/setstate、get_state/set_state；只查下表用到的状态操作 | 保存已经推进的随机流，而非仅保存 seed |

### 操作步骤

今天只在临时目录检查 40 步与 20+20 步恢复，使用固定的 C=32、context=16、batch=4、CPU FP32、dropout=0 小模型和教学 token。它不依赖 `mini_gpt_bpe.yaml` 或 `outputs/mini_gpt_bpe/last.pt`；正式 BPE 训练在 Day 5 开始。

#### 第一步：在训练入口加入 resume 参数和完整 checkpoint 分支

修改 `scripts/train_gpt.py::parse_args`，新增 `--resume`（type=Path，默认 None）。修改本文件 main()：在验证完成、更新 best 与日志累计值后，按下一步字段表构造完整 checkpoint，再调用 `src/petvision/utils/checkpoint.py` 已有的 save_checkpoint；在 main() 的对象构造阶段增加 resume 分支，先调用 load_checkpoint 核对配置，再创建并恢复模型、optimizer 和随机状态，最后进入统一循环。utils/checkpoint.py 的 save_checkpoint / load_checkpoint 继续只负责字典读写及 CPU、weights_only=True 加载，不把模型选择或训练循环放进去。

选择完成第 K 次 optimizer.step 后保存，记录 step=K。恢复后第一次更新应是 K+1，不能重复 K，也不能从 K+2 开始。

```text
连续：初始化 → 更新 1..40
中断：初始化 → 更新 1..20 → 保存
恢复：新建对象 → 加载状态 → 更新 21..40
```

#### 第二步：保存并恢复模型、优化器、日志与随机状态

模型参数决定当前预测；optimizer 的动量/矩估计决定下一步如何更新；scheduler 决定学习率；随机状态决定下一批采样和 dropout。tokenizer 与 split 标识决定 ID 和数据含义。

这里 RNG 是随机数生成器。seed 用于设置随机序列的起点，state 记录已经抽取若干次之后的当前位置；恢复到第 20 步时只重新设 seed，会从头取随机数，不能接上第 21 步。AdamW 的 exp_avg 是过去梯度的一阶矩估计，exp_avg_sq 是过去梯度平方的二阶矩估计；即使模型权重相同，丢失这些状态也可能让下一次更新不同。scheduler 是学习率调度器，GradScaler 用于某些混合精度训练中的梯度缩放；当前 FP32 小检查未启用它们，不需要为本步额外实现。

在第六周推理 checkpoint 的基础上增加这些字段：

| 字段 | 要保存和恢复的内容 |
| --- | --- |
| model_state / model_config | 参数、buffer、完整模型结构和词表大小 |
| optimizer_state | AdamW 的 step、动量和二阶矩等状态 |
| step / config | 已完成更新次数、累计目标与实际训练设置 |
| best_val_loss / best_step / 日志累计值 | 历史最佳验证结果、已处理 token/bytes 数、已记录训练秒数，以及未完成统计区间的 NLL 与目标数 |
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

采用与第二周一致的字典/Tensor 格式；NumPy 随机状态里的 ndarray 先转换成普通列表或 Tensor，恢复时再还原为所用 NumPy RNG 的状态格式，使本项目 checkpoint 可以按 `weights_only=True` 加载，相关格式见 [PyTorch 序列化说明](https://docs.pytorch.org/docs/stable/notes/serialization.html)。恢复时还原保存时已推进的具体随机状态。

只有模型参数相同，下一步训练仍可能不同。恢复流程应先构造模型/优化器并加载，再恢复随机状态，最后取下一批数据。

恢复历史 best_val_loss 后再判断新结果是否改善，不能重置为无穷大后覆盖更好的 best；沿用同一输出目录并保留原 best 文件。history 只追加恢复点之后的新验证行，若日志已有比 checkpoint.step 更晚的记录，先归档冲突记录，不能混入本轮。

配置核对也在恢复分支实际执行：模型结构、tokenizer / split hash、batch、context、优化器类型与超参数、验证间隔必须与保存时一致；允许提高 training.steps 这一累计目标，不能靠修改 YAML 学习率却被 optimizer_state 静默覆盖。当前等价续训检查使用同一 device / dtype。best.pt 的 step 及指标应与 last 中保存的 best_step / best_val_loss 对应；从更早备份恢复且目录里已有后续 best 时，应连同冲突日志归档后处理，不能把“未来的 best”混进这条恢复路径。

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

在编写对照测试之前，先把恢复分支按下列顺序接通：

1. 读取 resume 文件，核对 tokenizer、split hash、模型结构和训练设置，再读 completed_steps；配置的 steps 是累计目标，不是额外训练次数。
2. 按保存结构新建模型，加载 model_state 并迁移到目标设备。确认 `model.head.weight is model.token_embedding.weight` 仍为 True；权重共享由 MiniGPT 构造函数建立，load_state_dict 使用默认 assign=False。随后创建绑定这个新模型参数的 AdamW，再加载 optimizer_state；不能复用绑定旧模型的优化器。
3. 恢复 best_val_loss、best_step、训练统计累计值和日志进度，核对现有 history 没有超出该 checkpoint 的记录；冲突按前文规则归档处理。
4. 创建训练采样 Generator，完成其他对象初始化后，最后恢复它和实际使用的各随机流。恢复之后直接进入下一次实际采样，不额外抽样打印“看看随机数”。
5. 从 completed_steps+1 开始更新；如果已达到累计目标就退出训练。首次恢复更新后确认 step 恰好加一，日志仍按原验证间隔追加。
6. 恢复流程接入后，再编写下方 test_resume_matches_continuous；测试通过替换 parse_args 的返回值调用 train_gpt.main()，实际执行它的从零 / resume 分支与 save_checkpoint / load_checkpoint，不在测试中另写一套恢复模型和 RNG 的业务逻辑。

保存时统一使用已完成更新后的状态：若本步刚完成验证，先把该行日志写入、更新 best 值并清零区间累计，再写 checkpoint；若在验证间隔中间保存，区间 NLL / 目标数保持实际累计值。恢复后使用相同顺序，这样不会把上个区间重复统计。

因此今天增加完整恢复状态时，要把第六周循环里的 checkpoint 调用移到区间累计清零之后，再构造完整字典。资源统计也区分本进程 segment_seconds 与各段累计 train_seconds，恢复时接上 checkpoint 保存的累计值；若只测了恢复后的这一段，报告为该段耗时，不能标成完整训练时间。峰值显存记录各段峰值，汇总取最大值，不把不同进程的峰值相加。

##### tests/test_resume.py

创建 `tests/test_resume.py` 并定义 `test_resume_matches_continuous(tmp_path,monkeypatch)`。用 monkeypatch.syspath_prepend 将项目 scripts 目录临时加入搜索路径，然后导入 train_gpt；MiniGPT、sample_batch、load_checkpoint 从 petvision 对应模块导入。测试内使用 C=32、context=16、batch=4、CPU FP32、dropout=0 的临时配置；将语料读取替换为固定教学 token 数据，将配置、history、checkpoint 输出都指向 tmp_path。连续路径让 main() 跑到 40 步，中断路径先跑 20 步，再传入 --resume 对应的临时 last 跑到累计 40 步。此测试不读取正式 BPE 配置或输出文件。

使用 CPU FP32、dropout=0 的小模型，保留相同初始状态和数据。两条运行到 40 步后比较全部模型参数、optimizer step/状态、最后 loss，以及恢复后第一批的采样起点。

只比较最后 loss 很接近还不够，错误的批次或 optimizer 状态也可能偶然给出相近 loss。

对照程序按以下顺序组织：

1. 保存一份共同初始权重；连续路径和中断路径各自新建模型、AdamW 与 Generator，加载这份初值，使用同一教学 token 数据。
2. 连续路径运行 1..40，记录第 21 步实际采样起点；中断路径运行 1..20 后保存完整 last。两条路径使用同一验证时点，验证前后 Generator 状态不变。
3. 再新建模型、优化器与 Generator，加载第 20 步状态；恢复后第一次实际采样应与连续路径第 21 步一致。不要为了查看下一批而额外调用一次采样。
4. 两条路径完成 40 步后，在同一固定 batch、eval 下比较 loss；逐参数求最大绝对差。optimizer 按对应参数检查 step、exp_avg、exp_avg_sq，整数进度完全相同，浮点差小于 `1e-6`。
5. 随机状态另做短往返：保存后分别抽取 Python/NumPy/Torch 数值，恢复后重新抽取，应得到同一串；这项独立检查完成后重置测试状态，再运行训练对照。CPU 小检查不要求 CUDA 才能执行。

sample_batch 当前只返回 x/y，没有返回起点。测试中若要核对起点，可在实际采样之前 clone Generator 状态到一个独立探针 Generator，用相同的 randint 参数从探针取得预期起点，再正常调用 sample_batch；探针不推进真实训练 Generator。同时比较两条路径实际取得的 x/y。也可仅在测试辅助函数中返回起点，不必为打印多消耗一批真实训练数据。

恢复到 40 步之前先保留第 20 步 last 的副本。在同一测试的独立临时目录补一个停止边界：用这个副本恢复，累计目标也设为 20，断言不再调用 train_step，history 和 checkpoint 不被改写。steps 是累计目标：last 已完成 500 步、目标 1000 时再更新 500 步，目标仍为 500 时直接结束。验证不能消费训练采样随机流，恢复后核对 step 与下一批位置。

### 运行与自查

```bash
python -m pytest tests/test_resume.py -q
```

检查应自行创建小模型和教学 token 数据，并在 pytest 临时目录保存状态。输出或在断言失败时显示下一批采样起点、最大参数差、optimizer 状态差和最终 loss 差；目标为恢复路径与连续路径一致，起始容差 `1e-6`。今天不运行尚未创建的正式 BPE 配置。

- [ ] 模型、优化器、进度、随机状态和 tokenizer 均得到恢复。
- [ ] 连续与中断小规模结果对照通过，未发生重复/跳过更新。
- [ ] 能解释为什么只加载模型参数不足以等价续训。

<a id="day-5"></a>

## Day 5：BPE GPT 小预算训练与编码比较

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度与 tokenizer（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | tokenization 对 PPL 的影响 | 分开报告 token 级指标 |
| [第六周指南](week6.md) | 原文边界、训练和验证口径 | 复用相同原文 |

### 操作步骤

#### 第一步：完成 tokenizer 分派、BPE 配置和新模型构造

读取 bpe_tokenizer.json 的实际 V，据此新建 embedding、输出 head 和整个小 GPT。字符 ID 与 BPE ID 含义不同，不能直接把字符模型的同号行解释为同一内容。先完成本日的加载、统计和保存逻辑，正式从零训练按本日末尾顺序执行。

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

同时改造生成后的解码分支：字符 tokenizer 继续 decode(all_ids)；BPE 用 vocab 中的 bytes 拼接完整输出，严格解码失败时按 Day 2 的展示规则处理，并记录 decode_ok、raw_bytes_hex 和原始 IDs。不能仍无条件调用严格 decode 后在无效 UTF-8 处终止整个生成脚本。严格往返测试显式调用 `tokenizer.decode(ids, errors="strict")`，不经过展示分支。若统计替换字符数，注明这是展示文本中的 U+FFFD 数量，并不能准确代表无效字节数量。

#### 第二步：在训练日志累计输入 token 数与 UTF-8 字节数

context=128 在字符模型中约为 128 个已编码字符，在 BPE 中可能覆盖更多原文。报告实际 token 数与 UTF-8 字节数，解释比较范围。

在 scripts/train_gpt.py 的 main 中，与区间 NLL 统计一起累计每个实际训练批次的输入 token / bytes，并写入 run_summary。恢复训练时将这两个累计值与其他训练统计一起保存、恢复，不从本进程重新计零。

累计训练输入 token 数按每批 `x.numel()` 统计；BPE 的处理 bytes 数按输入 ID 对应的 vocab bytes 长度求和，重复采样重复计数，不包括只作为目标而未进入 x 的额外 token。静态训练文件大小、累计处理量和生成输出大小分列，避免把整个语料长度当作本次模型实际看过的字节数。

#### 第三步：实现 compare_tokenizers，编码同一原文并保存两行结果

创建 `scripts/compare_tokenizers.py`，顶层导入 Path、csv、hashlib、time，以及 `from petvision.models.llm.tokenizer import load_tokenizer`。main 读取第六周同一份 val.txt 和两份已保存的 tokenizer；文件末尾加入口保护，以下编码、计时和 CSV 保存都由该入口执行。

```text
固定同一 val 原文 → 字符 encode → token_count / encode_time
固定同一 val 原文 → BPE encode  → token_count / encode_time
```

先预热再计时，重复少量次取平均；记录原文 hash、字节数和 OOV。不能对不同长度的输入计时后直接比较快慢。

两种 tokenizer 先加载完成，固定同一份 val 字符串，各预热一次，再分别用 `time.perf_counter()` 测 3 次 encode、取均值。CSV 每行保存 tokenizer_hash、原文hash、chars、utf8_bytes、token_count、oov_count、mean_encode_seconds、repeats=3。两行的原文 hash/字符数/字节数应一致，token 数允许不同；字符 OOV 如实记录，BPE 对合法 UTF-8 输入不需要 UNK。

将两行实际结果用 csv.DictWriter 保存到 `experiments/mini_gpt/week7/tokenizer_comparison.csv`，创建父目录并写入表头；读取后核对两个 tokenizer 标识、相同原文 hash 和实际行数。这个文件记录编码实验，不能用生成耗时填入 mean_encode_seconds。

#### 第四步：保存 BPE 曲线与生成耗时，分别报告两种词表的 loss

先把本步的保存和计时逻辑接入脚本，再按下文顺序启动正式训练。训练完成后读取本次 history、run_summary 和生成记录，整理下面的结果。

BPE 验证 loss 单独报告，不能因数字低于字符 loss 就断言更好。比较生成耗时时注明同样 prompt、生成 token 数及实际输出字节数；token/s 依赖 tokenizer。

本周 BPE 曲线由 train_gpt.py 读取本次 history 后传给 plot_lm_history，保存为 `experiments/mini_gpt/week7/runs/bpe/figures/loss_curves.png`；run_summary.json 同目录保存训练时间、峰值显存和累计输入 token / bytes。生成计时在模型加载、prompt 编码完成后开始，GPU 前后同步，记录整个自回归循环的 generate_seconds；输出字节数由实际输出计算，不能用 max_new_tokens 代替。字符与 BPE 样例均用同一字段定义。

<a id="bpe-resume-order"></a>
正式 BPE 训练按以下顺序执行，具体命令见本节下方：

1. 确认 Day 2 的 bpe_tokenizer.json 已保存且能加载，再于今天创建完整的 mini_gpt_bpe.yaml；累计目标写入 `training.steps`，起始为 2000。
2. 首次从零运行，不传 `--resume`；last 按已完成更新步数保存完整训练状态。
3. 只有该次运行确实中断、已产生可读 last，且 `0 < last.step < training.steps` 时，才用同一配置和该文件恢复。例如已完成 500、目标 2000，则继续 1500 步。
4. 若已完成目标步数，直接评估和生成；恢复入口读取后也应退出，不额外增加步数。Day 4 的小规模检查已经验证恢复逻辑，正常跑完 BPE 不需要再人为中断一次。

恢复前核对 tokenizer、split、模型结构与训练设置；以原正式配置的累计目标继续，不能通过改词表或改模型结构套用旧 last。验证及最终测试沿用[第六周窗口规则](week6.md#eval-windows)，各自完整统计 N-1 个目标。

### 运行与自查

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

编码对照应读取相同 val 原文，保存到 `tokenizer_comparison.csv`；BPE 样例追加到本周 `samples.jsonl`。从零训练命令用于首次运行，恢复命令仅用于尚未达到累计目标的中断运行。

- [ ] BPE 模型与自己的词表配套，没有直接套用字符模型输出头。
- [ ] 跨 tokenizer 的比较使用相同原文，loss/PPL 分开报告。
- [ ] BPE 训练、曲线、编码结果和生成样例均来自实际运行。

<a id="day-6"></a>

## Day 6：最终语言评估与周总结

### 阅读范围

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [困惑度评估说明（中文译文）](https://hugging-face.cn/docs/transformers/perplexity) | 相同 tokenizer 下的比较口径 | 整理最终指标 |
| [总计划语言模型验收](StudyPlan.md#acceptance) | BPE、续训和样例要求 | 核查两周成果 |

### 操作步骤

为训练入口增加 `--evaluate-only --checkpoint --split`，三个冻结模型分别评估自己的测试原文。

#### 第一步：分别评估各自 tokenizer 的测试原文

继续修改 `scripts/train_gpt.py`，增加下文的 evaluate-only 分支，复用 `src/petvision/training/lm_trainer.py` 中的 evaluate_lm。

冻结字符 Bigram、字符 GPT、BPE GPT 的配置和 best，在相同原文 test 边界内分别编码评估。记录有效目标 token 数与窗口规则。

字符两组可直接比较 token loss；BPE 使用独立列或独立表，保留原文范围与词表信息。

`--evaluate-only` 分支在创建训练循环和写训练文件之前返回：按 checkpoint 重建模型与 tokenizer，只读取所选 split，切 eval 并关闭梯度，不更新参数。对长度为 N 的编码序列，用第六周的固定分窗方式统计 `N-1` 个有效目标，保留尾窗并按目标数汇总 NLL。打印模型、tokenizer hash、原文范围、目标数与 loss；测试结果整理到本周总结，不覆盖训练 history 或 checkpoint。

修改 `scripts/train_gpt.py::parse_args`：`--evaluate-only` 为 store_true，`--checkpoint` 为 Path，`--split` 限定 val/test；evaluate-only 时必须提供 checkpoint，且与 resume 互斥。先在临时小模型上检查这个分支返回完整目标计数、参数未变，正式运行前后 history 行数与 checkpoint 保持一致。这样下方三条命令确实只评估，不会先训练再评估。

建议给 --split 默认 val；未开启 evaluate-only 时传 --checkpoint 应报错，避免误以为它会恢复训练。只评估分支在 main 中先执行并 return，再进入新训练 / resume 分支。把本次真实结果保存到所选配置的 `experiment_dir/{split}_metrics.json`，字段包含 model、checkpoint、checkpoint_step、tokenizer_hash、split_hash、split、n_target_tokens、loss 和窗口规则。三组分别写自己的目录，再在本周 README 链接它们；结果保存只涉及评估文件，不改训练 config、history 或权重。

完成 train_gpt.py 的分支、参数校验和保存逻辑后，再编写检查：

在 `tests/test_language_model.py` 新增 `test_lm_evaluate_only_does_not_train(tmp_path,monkeypatch)`：创建临时模型 / tokenizer / checkpoint 和长度 N=8 的教学 split，用替换 parse_args 的方式调用 train_gpt.main() 的 evaluate-only 分支；断言目标数为 7，训练 history 和 checkpoint 文件字节未变，并将 train_step 替换为 pytest.fail，验证该分支不会调用参数更新。通过后再执行下面三条正式测试命令。

#### 第二步：汇总测试、编码、采样和续训结果，列出待补项

在本周 README 的周总结中依次完成以下四项，每项先取实际产物再写结论：

1. 字符基线：读回字符 Bigram / GPT 的 test_metrics，核对 tokenizer hash 和有效目标数相同，再并列报告 loss；BPE 结果单列，写自己的目标数与词表标识。
2. BPE：引用严格往返检查的命令与结果，再读取 tokenizer_comparison.csv，核对两行使用同一原文后，报告 token 数和编码耗时差异。
3. 采样：从 samples.jsonl 选取本次指定的两组记录，保留完整设置和原始输出，描述重复、连贯性等观察；两项采样参数同时变化时，不将差异只归因于其中一项。
4. 续训：填写 CPU 小对照的配置、实际检查命令、最大参数差、优化器状态及第 21 步批次是否一致。正式 BPE 从未中断也可以如实记录，不能虚构一次正式恢复经历。

逐项加上实际配置或结果链接；缺一项就标记待补，不把代码文件存在等同于已经运行通过。

#### 第三步：在短输入上重算 BPE、采样，并解释续训状态

在草稿中重新统计 abab / aaaa 的 pair 并执行一次非重叠合并，实际核对前面的结果；再对 `[2,1,0,-1]` 依次做温度缩放、top-k 和 softmax，核对保留项与概率。最后按续训状态表逐项说明作用，重点结合第 21 步批次和参数差解释模型、optimizer、Generator 的影响。将运行结果和仍答不出的点写入本周 README。

如果 Unicode 往返、续训一致性或指标口径仍不清楚，记录具体失败输入和下一步。更多语料、上下文长度和 tokenizer 产品化放入后续路线。

### 运行与自查

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

保留 tokenizer 文件、续训误差、三组测试结果和失败样例，供第八周总结。更多上下文长度与语料实验另行安排。

通过后进入[第八周](week8.md)。
