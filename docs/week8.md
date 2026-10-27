# 第八周逐日学习指南：CLIP、LLM / VLM 推理与八周总结

> 更新日期：2026-10-27<br>
> 对应计划：[总学习计划](StudyPlan.md#stage-4)  
> 时间：6 个学习日，每天 4 小时明确任务 + 1 小时机动，共 30 小时；第 7 天休息。Day 1 为本周第一个学习日，可从任意星期开始。  
> 本周成果：本地完成 CLIP 零样本任务，云端运行小型 LLM 与 VLM，保存至少 20 张图×每图2问的完整结果、人工判定和结构图，整理最终报告。  
> 环境：CLIP 可在本地 Conda `petvision` 运行；LLM/VLM 在云端 48GB GPU 的独立 Conda `petvision-vlm` 运行，实际依赖和精度以模型卡与兼容版本为准。

## 怎么使用这份指南

先确认[上一周指南](week7.md)的基础关卡已通过。前七周基础和核心模型应已验收；本周主要阅读与推理，完整模型微调、Encoder–Decoder 和 Gradio 系统保留到八周之后。

本指南按学习日写出环境准备、文件创建、操作步骤、运行命令和自查方法，跟做时从当天内容顺序往下读即可。[本周 README](../experiments/vlm_qa/week8/README.md) 保存实际环境、复现命令索引、结果链接和周总结。源码、配置、数据与结果按学习日逐步创建；下面的命令在完成相应实现后执行，代码由你亲手完成。

每天按“读指定部分 → 写代码 → 运行检查 → 解释结果”执行。检查通过后再扩大数据或训练预算；日常通过代码、输出和解释验收，周末集中更新本周 README。学习时间包含机动时间；训练等待与阅读重叠时只计一次。

文件路径均相对于项目根目录；可复用源码放在 `src/petvision/`。新增源码子目录时，在每层包目录放入空的 `__init__.py`；已有目录和文件继续修改，保留此前通过检查的内容。`scripts/` 放运行入口，`tests/` 放检查，结果由脚本运行后生成。

## 本周目录蓝图

以下为本周逐步形成的文件，注释标明创建或修改时间。`README.md` 已提供；其余文件和结果在执行对应任务时产生。

```text
PetVisionLab/
├── environment-vlm.yml                     # Day 2：云端独立环境定义
├── configs/
│   ├── cv/clip_zeroshot.yaml               # Day 1：模型、类别模板、固定图像
│   ├── llm/llm_inference.yaml              # Day 2：LLM/revision/生成参数
│   └── vlm/vlm_inference.yaml              # Day 3：VLM/revision/图像与生成
├── scripts/
│   ├── run_clip.py                        # Day 1：零样本分类
│   ├── infer_llm.py                       # Day 2：文本推理
│   └── infer_vlm.py                       # Day 3–4：图像+问题、批量保存
├── datasets/raw/vqa/                       # Day 3–4：本地图片，不上传大文件
├── experiments/vlm_qa/week8/
│   ├── README.md                         # 已提供
│   ├── clip_subset.json                 # Day 1：固定的 200 个全局图像索引
│   ├── clip_results.csv                  # Day 1：零样本结果
│   ├── model_manifest.json               # Day 1–3：模型/代码/依赖版本
│   ├── environments/{local,cloud}/       # Day 1–2：分别保存两端环境快照
│   ├── llm_outputs.jsonl                  # Day 2：真实文本生成
│   ├── images_manifest.csv               # Day 4：20+ 图片与来源
│   ├── questions.jsonl                    # Day 4：40+ 问题和预先参考标准
│   ├── vlm_outputs.jsonl                  # Day 4：全部原始回答与运行信息
│   ├── judgments.csv                      # Day 4：人工判定及错误类型
│   ├── figures/clip_mistakes.png            # Day 1：固定子集的错误样例（有错分时）
│   └── tensor_flow.md                    # Day 3/5：实际结构与关键形状
└── docs/final_report.md                    # Day 5–6：八周报告与未完成事项
```

## 固定条件与接口约定

首选入门模型为 CLIP `openai/clip-vit-base-patch32`、LLM `HuggingFaceTB/SmolLM2-360M-Instruct`、VLM `HuggingFaceTB/SmolVLM-256M-Instruct`，均先按官方模型卡验证一次最小推理，再记录实际 commit/revision、模型类与依赖版本。它们是小模型起点，不要求耗尽 48GB 显存。若兼容性问题需更换模型，保留变更原因及新版本记录。

CLIP 默认对第三周 CIFAR 验证集按种子 42 每类选 20 张，共 200 张，以固定英文类别模板做零样本分类，使用模型自己的 processor。结果单列为预训练零样本条件。LLM/VLM 首次用 batch=1、greedy（do_sample=False）、max_new_tokens=128、固定 prompt 与精度；GPU 支持时可用 BF16，否则明确记录 FP32/FP16。VQA 在推理前写好至少 20 张图、每图 2 个问题与判定标准，覆盖类别、颜色、数量、位置和信息不足；每个问题独立上下文，不让前一题答案进入下一题。所选 SmolVLM 模型卡标注主要语言为英语，本轮优先用英文问题与判定标准；如增加中文题，单独报告语言条件和结果，避免把语言能力差异全部归因为视觉理解。

## 六天总览

<a id="inference-terms"></a>
本周先区分下列对象和记录，后面的脚本会分别创建或保存它们：

| 名称 | 本项目中的含义 |
| --- | --- |
| CLIP / zero-shot | 图像和文本分别编码后比较相似度；零样本指本次不使用 CIFAR 标签训练这些预训练权重，而用类别文本直接分类 |
| LLM / VLM / VQA | 大语言模型 / 视觉语言模型 / 视觉问答任务；本周分别做纯文本和带真实图片的推理 |
| tokenizer / processor | tokenizer 把文字编码为 ID；processor 负责模型所需的图像预处理，并可统一组织图文输入 |
| prompt / Chat Template | 给模型的提示内容 / 把角色、内容和控制标记排成该模型所需格式的模板；模板不是参考答案 |
| model ID / revision | 模型仓库名称 / 该仓库的具体版本；固定完整 commit SHA 是为了再次加载同一版模型及 processor |
| FP32 / FP16 / BF16 | float32 / float16 / bfloat16 浮点格式；后两者都是 16 位但数值范围和精度不同，需按模型与设备支持选择 |
| greedy / do_sample=False | 每一步选得分最高的下一个 token，不做随机采样；不代表输出一定正确 |
| max_new_tokens | 最多新增多少个 token，不含输入 prompt；遇到结束 token 时可能提前停止 |
| image_id / qid / run_id | 图片编号 / 问题编号 / 一整轮推理的编号；同一轮的所有题共享 run_id，重跑才换一个 |

| 学习日 | 主题 | 明确任务 | 机动时间 | 核心交付 |
| --- | --- | --- | --- | --- |
| Day 1 | CLIP 图文相似度与零样本分类 | 4h | 1h | 固定 200 张图的零样本结果 |
| Day 2 | 云端 Conda 环境与小型 LLM | 4h | 1h | LLM 脚本、Chat Template 与版本记录 |
| Day 3 | 真正的图像输入与 VLM 张量流 | 4h | 1h | 单图问答与实际模块结构 |
| Day 4 | 20 张图、40 个问题的完整评估 | 4h | 1h | 原始回答、预先标准和人工判定 |
| Day 5 | 跨模块复盘与最终报告初稿 | 4h | 1h | 有证据的八周报告 |
| Day 6 | 复现抽查、解释与最终整理 | 4h | 1h | 最终报告和下一轮学习清单 |
| Day 7 | 休息 | 0h | 0h | — |
| 合计 | — | 24h | 6h | 30h |

<a id="day-1"></a>
## Day 1：CLIP 图文相似度与零样本分类

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [CLIP 原论文](https://arxiv.org/abs/2103.00020) | 双编码器、图文对比与零样本分类 | 画出图像/文本 embedding |
| [CLIP 官方实现](https://github.com/openai/CLIP) | zero-shot 示例、特征归一化与相似度 | 理解候选类别打分 |
| [CLIP 模型卡](https://huggingface.co/openai/clip-vit-base-patch32) | 模型加载与对应预处理 | 运行固定模型/revision |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读 CLIP 与模型卡 |
| 2h | 实现固定样本的零样本分类 |
| 1h | 保存结果与失败样例并解释分数 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `configs/cv/clip_zeroshot.yaml` 和 `scripts/run_clip.py`。沿用 CIFAR 验证索引，每类按固定种子选 20 张并保存实际索引；加载时使用 CLIP 对应 processor，不复用分类 CNN 的归一化。

2. 固定 10 个英文类别与模板，例如 a photo of a {class}。获取图像与文本特征，按参考实现归一化后计算相似度并得到预测，打印实际 embedding 形状。

3. 在 eval 和关闭梯度下跑完 200 张，保存 image_id、真实类、预测类、类别得分及配置标识到 clip_results.csv，报告该固定子集 accuracy 和失败样例。

4. 说明 softmax 得分依赖候选类别与模板；CLIP 预训练零样本结果单列，不与第五周从零训练预算混为同一种设置。按下面步骤准备依赖、固定权重 revision 并保存本地环境。

### 分步操作与自查

<a id="clip-environment"></a>
#### 第一步：补充本地依赖并建立 CLIP 配置

在本地项目根目录激活已经验证过的环境，先检查现有 PyTorch，再按 [Transformers 安装说明（官方中文）](https://huggingface.co/docs/transformers/zh/installation)和 [CLIP 模型卡](https://huggingface.co/openai/clip-vit-base-patch32)补充推理依赖：

```bash
conda activate petvision
python -c "import sys, torch, torchvision, PIL; print(sys.executable); print(torch.__version__); print(torchvision.__version__)"
python -m pip install transformers accelerate huggingface_hub
python -m pip check
```

这是首次补充依赖的起始命令；模型卡若指定版本约束，安装时使用对应约束。已有兼容组合时直接复用，不重复升级。torch / torchvision 沿用已经通过 GPU 检查的配套版本，Pillow 用于读取图片；安装后打印 transformers 与 huggingface_hub 的实际版本并记录。

创建 `scripts/run_clip.py`，用 argparse 接收必填的 `--config` 路径，在 main 中用 yaml.safe_load 读取；文件末尾加入口保护，后面第二至第五步依次接入这个 main。再创建 `configs/cv/clip_zeroshot.yaml`，字段由该脚本读取：

```yaml
model:
  id: openai/clip-vit-base-patch32
  revision: null
data:
  root: datasets/raw
  split_path: experiments/cnn_vs_resnet/week3/cifar10_split.json
  subset_path: experiments/vlm_qa/week8/clip_subset.json
  samples_per_class: 20
  seed: 42
  batch_size: 16
  classes: [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
prompt_template: "a photo of a {class}"
inference:
  device: auto
  dtype: float32
output_file: experiments/vlm_qa/week8/clip_results.csv
```

`revision: null` 只用于首次确定版本。先通过 [Hub 的 model_info 接口](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)查询当前完整 commit SHA：

```python
from huggingface_hub import HfApi

model_id = "openai/clip-vit-base-patch32"
revision = HfApi().model_info(model_id).sha
print(revision)
```

把这次 SHA 填回配置，再将同一个 `revision` 同时传给模型与 processor 的 `from_pretrained`。LLM/VLM 也按同样方法固定各自仓库版本；后续复现读取已保存 SHA，不每次重新查询最新版本。配置里的 `id`、`dtype` 等字段需要入口解释，不能把整个 YAML 直接当作模型加载参数。

最小前向验证后保存实际环境。环境检查脚本先创建输出目录，后续导出再写入该目录：

```bash
python scripts/check_environment.py --output-dir experiments/vlm_qa/week8/environments/local
conda list > experiments/vlm_qa/week8/environments/local/conda-packages.txt
conda list --explicit > experiments/vlm_qa/week8/environments/local/conda-explicit.txt
python -m pip freeze > experiments/vlm_qa/week8/environments/local/pip-freeze.txt
```

创建 `experiments/vlm_qa/week8/model_manifest.json`，以 clip、llm、vlm 为键分别保存三个条目，记录模型类、SHA、processor、精度和对应环境目录。先加入 CLIP；后两天先读取已有 JSON、只更新各自条目再写回，避免保存 LLM 时覆盖 CLIP 记录。采用第二周的 JSON 读写方式，父目录先创建，不写入尚未运行模型的虚构结果。

三个推理脚本都沿用第二周的 argparse / yaml.safe_load：顶层导入依赖，在 main 中读取 --config 为 cfg，再依次创建模型、准备输入、推理和保存。下面的配置 id / revision 用于 from_pretrained，device / dtype 需要先变为 Torch 对象，output_file 只传给结果写入逻辑，不能把 YAML 整体展开给模型构造函数。

#### 第二步：先固定图片、类别与模板

从已保存 CIFAR 验证索引中，每类按 seed=42 选 20 张。保存实际选择的 index；反复运行直接读取它们，避免样本变化造成指标变化。

先加载 `train=True` 的原始 CIFAR 图片及 targets，按第三周 val_indices 筛出候选索引；每类从候选中用固定 Generator 无放回选 20 个，保存到配置的 subset_path，记录 dataset、seed、原划分标识和全局 indices。断言 200 个索引无重复、全部属于 val、每类恰好 20；已有文件直接读取并核对条件。取图时使用这些全局索引，不能把 Subset 内的局部编号误当原始 image_id。

10 个类别各对应一句固定英文模板，例如 `a photo of a cat`。类别顺序与输出列顺序一致，先打印类别列表核对。

<a id="clip-inputs"></a>
#### 第三步：创建 CLIP 对象并处理图文输入

图像由 CLIP processor 调整尺寸和归一化，文本由配套 tokenizer 编码。本周使用预训练模型配套的 processor；第三、五周的 v2 分类预处理不适用于直接生成它的输入。创建原始 CIFAR10 数据集时设置 `transform=None`，取出的 RGB PIL 图片直接传给 processor，不先经过 v2.ToDtype / v2.Normalize，也不添加训练用随机增强，避免重复缩放和归一化。Day 3 的 VLM 同样把 RGB PIL 图片交给其配套 processor，由它按模型要求处理。

```text
图片批次 → image encoder → image features [B,D]
类别文本 → text encoder  → text features  [K,D]
归一化后相似度                           [B,K]
```

这里 B 是本批图片张数，K 是候选类别文本数量（本题为 10），D 是模型输出的共同特征长度，由实际模型决定。K 在此处既不是卷积核大小，也不是 Attention 的 key 张量。`[B,K]` 中第 b 行、第 k 列就是“第 b 张图片与第 k 个类别文本”的匹配分数。

先在 run_clip.py 顶层导入 torch、CLIPModel、CLIPProcessor。配置中的 revision 已填入真实 SHA 后，在 main 内加载：

```python
model_id = cfg["model"]["id"]
revision = cfg["model"]["revision"]
if not isinstance(revision, str) or not revision:
    raise ValueError("请先把模型完整 commit SHA 填入 revision")
device_name = cfg["inference"]["device"]
if device_name == "auto":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
dtype = {"float32": torch.float32, "float16": torch.float16,
         "bfloat16": torch.bfloat16}[cfg["inference"]["dtype"]]
processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
model = CLIPModel.from_pretrained(model_id, revision=revision).to(device=device, dtype=dtype).eval()
```

取原始 CIFAR 图片时用 `CIFAR10(root=..., train=True, transform=None)`，从固定全局索引读取 PIL 图片，按配置 batch_size 组成 images 列表。类别文本 texts 用 `[cfg["prompt_template"].format(**{"class": name}) for name in cfg["data"]["classes"]]` 建立；10 句文本对应 10 个候选类，不是逐张图片的说明。

在每个图片批次中执行以下最小路径；首轮 FP32，processor 返回的整数文本字段只迁移设备：

```python
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
inputs = inputs.to(device)
inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
with torch.no_grad():
    outputs = model(**inputs)
scores = outputs.logits_per_image
predicted = scores.argmax(dim=-1)
print(outputs.image_embeds.shape, outputs.text_embeds.shape, scores.shape)
```

本路径的 scores 为模型缩放后的 clip_logits，形状 `[B,10]`；输出图像 / 文本特征用于下方余弦相似度核对。先跑一个批次核对，再遍历完整 200 张并写结果。不要把普通 DataLoader 默认的 Tensor 拼批方式用于未转换的 PIL 图片，直接按保存索引建立图片列表即可。

#### 第四步：从相似度得到分类预测

沿 K 维 argmax 得到类别。需要展示分数时，在固定候选类别上 softmax；这些分数会受模板和候选集合影响，不直接当校准后的正确率概率。

若与 CLIP 模型返回的分类 logits 对齐，还要应用模型学习到的 logit scale；只做归一化特征点积得到的是余弦相似度。正的缩放不改变 argmax，但会改变 softmax 分数，保存结果时写清实际使用的是哪种分数。

先用两维特征核对方向：图像特征 `[[3,4],[0,2]]`，文本特征 `[[1,0],[0,1]]`，逐行 L2 归一化后相似度应为 `[[0.6,0.8],[0,1]]`，两个预测均为类别 1。真实模型中取出实际特征 Tensor，再计算 `image_unit @ text_unit.T`，输出 `[B,10]`；检查最后一维对应 classes 的顺序，而不是沿 batch 做 argmax。

这里 L2 归一化是“向量除以自身长度”，长度为各分量平方和的平方根：`[3,4]` 的长度是 `sqrt(3²+4²)=5`，归一化后为 `[0.6,0.8]`。单位向量之间的点积就是余弦相似度。它与前几周对像素做 `(x-mean)/std` 的 Normalize 是两种不同操作，不能相互替代。

#### 第五步：保存一次完整子集结果

在 `scripts/run_clip.py` 中读取 cfg["output_file"]，将结果保存到 `experiments/vlm_qa/week8/clip_results.csv`；父目录先创建，后面的列和完整性检查针对这份文件。

clip_results.csv 每张一行，含 image_id、真实类、预测类、分数和模型标识；整体 accuracy 从全部 200 张统计。报告预训练零样本条件，与第五周从零训练结果分开。

类别得分以 `score_airplane ... score_truck` 十列按 classes 顺序保存，并带 score_type，明确是 cosine_similarity 还是应用 logit scale 后的 clip_logits；预测和 accuracy 使用同一套得分。用第一周 csv.DictWriter 写表，再读回核对 200 个 image_id 恰好等于固定子集，正确数/200 与终端 accuracy 一致。

构造 CSV 行时，image_id 使用原始 CIFAR 全局索引，真实类取同一索引的 targets，预测类按配置 classes 顺序解释；先断言配置 classes 等于数据集 classes，避免同名类别顺序错位。每行加入 model_id、revision、prompt_template 和 config 标识，model_manifest 的 clip 条目保留本次完整配置。

从这次错误预测中最多选 8 张原始 PIL 图片，用第二周 subplots 方法标注真实类 / 预测类，保存到 `experiments/vlm_qa/week8/figures/clip_mistakes.png`。直接显示原图，不对 PIL 再逆归一化；没有错误时在 README 记录 0 个样例。样例仅用于解释，accuracy 始终从完整 200 张计算。

### 运行与验收

```bash
python scripts/run_clip.py --config configs/cv/clip_zeroshot.yaml
```

核对固定子集恰好 200 张、每类 20 张，`clip_results.csv` 每图一行。首次下载失败先看网络、缓存与完整错误；图片类型报错时检查传入的是否为原始 RGB 图片；输入正常但类别错位时检查 classes 顺序与 logits 列顺序。

- [ ] 图文输入、预处理和相似度流程正确，模型来源有记录。
- [ ] 固定类别、模板、样本索引和原始分数可复现。
- [ ] 能解释 embedding 相似度与候选类别分数的含义。

<a id="day-2"></a>
## Day 2：云端 Conda 环境与小型 LLM

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [SmolLM2 模型卡](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 官方最小加载与对话生成示例 | 先运行一个 prompt |
| [Hugging Face Chat Template（官方中文）](https://huggingface.co/docs/transformers/zh/chat_templating) | messages、特殊 token 与生成前缀 | 检查实际 input_ids |
| [第五周模型计时方法](week5.md#inference-timing)、[第六周 JSONL 写入](week6.md#jsonl-output) | 同步与 perf_counter；每行一个结果对象 | 把资源与完整回答接到推理循环 |
| [PyTorch 安装说明（中文译文）](https://pytorch.ac.cn/get-started/locally/)（[官方命令向导](https://pytorch.org/get-started/locally/)） | 云端驱动对应 wheel | 验证独立环境 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 建立独立云端环境并验证 GPU |
| 1h | 阅读模型卡与 Chat Template |
| 2h | 实现 LLM 推理、记录输入输出与资源 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 同步源码和配置到云端项目，按下面第一步创建独立 Conda `petvision-vlm` 并安装依赖；创建 `environment-vlm.yml`，将云端 GPU、驱动、显存与 pip/Conda 快照单独保存。

2. 创建 `configs/llm/llm_inference.yaml` 与 `scripts/infer_llm.py`，先加载模型和 tokenizer 并固定真实 revision；按当前兼容模型卡选择 API，不直接复制不匹配版本的示例。

3. 用 Chat Template 处理 5 个固定英文 prompt，打印输入 token 形状，生成时只解码新增 token。保存 messages、实际模板/输入标识、生成设置、原始回答与模型信息到 llm_outputs.jsonl。

4. 计时分开记录加载与生成，GPU 生成计时前后同步，记录峰值显存；画 `[B,T]→embedding→causal blocks→[B,T,V]` 并联系自己的 Mini-GPT。

### 分步操作与自查

<a id="cloud-environment"></a>
#### 第一步：在云端创建并验证独立环境

先通过云平台文件上传或已有同步工具，将 `src/、scripts/、configs/、pyproject.toml` 和本周实验目录传到云端项目根目录，目录层级保持一致；Day 3–4 再补入实际图片与问题清单。进入云端含 pyproject.toml 的目录后执行。首次创建环境：

```bash
conda create -n petvision-vlm -c conda-forge python=3.12 pip numpy matplotlib pyyaml pillow
conda activate petvision-vlm
python -c "import sys; print(sys.executable)"
nvidia-smi
```

已有同名环境时直接激活并核对依赖。解释器应来自云端 `petvision-vlm`；记录实际 GPU、驱动、显存，不能仅因租用规格写着 48GB 就省略检查。

再按顺序准备 PyTorch 与推理库：

1. 打开 [PyTorch 安装说明（中文译文）](https://pytorch.ac.cn/get-started/locally/)（[官方命令向导](https://pytorch.org/get-started/locally/)），按云端实际系统选择 Stable、Pip、Python 和兼容驱动的 CUDA 构建。
2. 在已激活环境中执行生成的配套 torch / torchvision 命令，开头使用 `python -m pip`，保留官方命令的版本与 index-url。
3. 依据 SmolLM2 / SmolVLM 模型卡的兼容要求，补充 transformers、accelerate 与 huggingface_hub。不要直接使用笔记本的同平台 explicit 清单重建另一平台的云端环境。

```bash
python -m pip install transformers accelerate huggingface_hub
python -m pip install -e . --no-deps
python -m pip check
python -c "import sys, torch, torchvision, transformers, petvision; print(sys.executable); print(torch.__version__); print(torchvision.__version__); print(transformers.__version__); print(petvision.__file__)"
python scripts/check_environment.py --output-dir experiments/vlm_qa/week8/environments/cloud
```

若模型卡要求固定版本，使用相应版本约束替换上面的起始安装命令。环境检查应能执行小型 CUDA 运算；故障时按第二周已经解释过的顺序检查当前解释器、CUDA 构建与驱动。先用标准 Attention 实现跑通最小样例；模型卡中可选的 Flash Attention 不是本轮前置依赖。

小模型可先用 FP32 推理；确认 GPU 支持 BF16 后，再显式选择 bfloat16 并记录。两种情况下都检查实际模型与浮点输入 dtype。加载耗时与生成耗时分别记录，不需要为了占满显存选择更大模型。

今天单 prompt 验证通过后，在云端项目根目录执行 `conda env export --no-builds > environment-vlm.yml`，删除导出文件中的本机绝对 prefix，保留完整 PyTorch 安装命令和实际 index-url。次日 VLM 若需要调整依赖，再更新对应快照并注明模型各自使用的版本。这个 YAML 仍是云端平台的依赖快照，不等于跨平台锁文件。随后在已经由环境检查脚本创建的 cloud 目录导出：

```bash
conda list > experiments/vlm_qa/week8/environments/cloud/conda-packages.txt
conda list --explicit > experiments/vlm_qa/week8/environments/cloud/conda-explicit.txt
python -m pip freeze > experiments/vlm_qa/week8/environments/cloud/pip-freeze.txt
```

本地 CLIP 与云端 LLM/VLM 使用不同快照目录；model manifest 中记录各自使用的环境目录，下载云端结果时也保持这两个目录分开。

<a id="llm-inference"></a>
#### 第二步：先创建配置，再加载 LLM 和 tokenizer

创建 `scripts/infer_llm.py`，沿用 Day 1 的 `--config` 参数解析、yaml.safe_load、main 与入口保护；后面的对象创建和五个 prompt 推理都接在该 main 中。再创建 `configs/llm/llm_inference.yaml`，按 Day 1 的 SHA 查询步骤固定本模型 revision。下面是首次配置模板，null 要在加载前替换：

```yaml
model:
  id: HuggingFaceTB/SmolLM2-360M-Instruct
  revision: null
data:
  batch_size: 1
  prompts:
    - "Explain gradient descent briefly."
    - "What is 7 multiplied by 8?"
    - "List three common image classification models."
    - "Reply with a JSON object whose only key is greeting and value is hello."
    - "Rewrite 'The model made a mistake' in simpler English."
inference:
  device: cuda
  dtype: float32
  do_sample: false
  max_new_tokens: 128
output_file: experiments/vlm_qa/week8/llm_outputs.jsonl
```

在 infer_llm.py 顶层导入 torch、AutoTokenizer、AutoModelForCausalLM，在 main 读出 cfg 后执行。device / dtype 的转换沿用 Day 1；这里云端配置明确使用 cuda，必须先通过环境检查：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = cfg["model"]["id"]
revision = cfg["model"]["revision"]
if not isinstance(revision, str) or not revision:
    raise ValueError("请先把模型完整 commit SHA 填入 revision")
device = torch.device(cfg["inference"]["device"])
dtype = {"float32": torch.float32, "float16": torch.float16,
         "bfloat16": torch.bfloat16}[cfg["inference"]["dtype"]]
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
model = model.to(device=device, dtype=dtype).eval()
```

from_pretrained 在本地缓存缺少该版本时获取权重 / tokenizer；两个对象必须使用同一个已保存 SHA。先用 FP32 跑通本节单 prompt，再按硬件能力调整精度并记录。

#### 第三步：看清 messages 如何变为 token

```python
messages = [{'role': 'user', 'content': 'Explain gradient descent briefly.'}]
# 使用该 tokenizer 的 Chat Template 得到模型输入
```

查看模板后的特殊 token、role 标记、生成前缀和 input_ids 长度。不能自行拼一个与模型训练格式无关的字符串后把效果差全部归因于模型。

先用 `apply_chat_template` 查看格式化文本，再使用模型要求的生成前缀编码。如果先得到文本、再单独调用 tokenizer，要按 [Chat Template 说明（官方中文）](https://huggingface.co/docs/transformers/zh/chat_templating)避免重复添加特殊 token；也可以直接让模板接口输出 token，具体参数与当前已固定版本一致。

先选一条完整路径运行，例如「模板转文本 → tokenizer 编码」：

```python
rendered_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
```

打印 rendered_prompt、input_ids.shape 与实际 ID；模板已经包含特殊 token，因此本路径编码时设置 add_special_tokens=False。若使用直接输出 token 的路径，则按固定版本模型卡取得模型要求的输入字典，不再重复 tokenize 一次。

逐个参数理解：`tokenize=False` 先返回格式化后的字符串，便于查看；`add_generation_prompt=True` 请求模板加入开始回答所需的提示部分，实际标记取决于该模型模板；`return_tensors="pt"` 让编码结果使用 PyTorch Tensor。inputs 通常含 input_ids（整数 token ID）和 attention_mask（区分有效输入与填充的位置，具体形状由接口决定）。后者不能直接当作第四周约定 True=屏蔽的手写布尔 Mask 使用，接口含义需分别核对。

每条 prompt 单独构造一轮 user 消息，不把上一题回答放进下一题。设置 greedy 时不需要 temperature/top-k；先用第一题跑通，再遍历 cfg["data"]["prompts"] 中的五题。

当前入口只实现单条逐题处理，在读取配置后校验 data.batch_size==1，max_new_tokens 为正整数；配置其他 batch 值时明确报错，不能仍逐条执行却报告为批量推理。device、dtype、do_sample、max_new_tokens 都必须用于实际对象或调用。本轮固定 do_sample=False；以后启用采样时，再补充种子和采样参数。

#### 第四步：调用 generate，再只解码新生成部分

因果模型 generate 的结果通常包含输入前缀；记录输入长度，再取新增 ID 解码。先检查返回结构，避免把原 prompt 重复计作模型回答。

本轮 batch=1。完成第三步的 inputs 后，实际调用生成：

```python
with torch.no_grad():
    generated = model.generate(
        **inputs,
        do_sample=cfg["inference"]["do_sample"],
        max_new_tokens=cfg["inference"]["max_new_tokens"],
    )
sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
```

然后按输入长度切出新增部分：

```python
input_length = inputs["input_ids"].shape[-1]
new_ids = sequences[:, input_length:]
answer = tokenizer.decode(new_ids[0], skip_special_tokens=True)
print("new token count:", new_ids.shape[-1])
print(answer)
```

核对 `sequences.shape[-1] == input_length + new_ids.shape[-1]`，新增数量不超过 max_new_tokens；提前 EOS 可以少于 128。保存原 messages、rendered_prompt、输入/新增 ID 和 answer；五个 prompt 各自重新构造消息，按第六周 JSONL 方法写五条正式结果。

写入路径使用 cfg["output_file"]，本题为 `experiments/vlm_qa/week8/llm_outputs.jsonl`。为本次完整运行生成一个 run_id，五条结果共用该标识；若追加保存多次运行，检查时先按 run_id 筛选本次五条，不能把暖机或重跑记录当成新增题目。

顶层导入 `from uuid import uuid4`，在正式五题循环之前生成一次 `run_id = uuid4().hex`；不要每题重新生成。给每题稳定的 prompt_id（如 llm_q1..llm_q5），record 保存 run_id、prompt_id、messages、rendered_prompt、input_ids、new_ids、raw_answer、model_id、revision、实际 inference 配置和资源测量值，再按第六周方法追加一行。Tensor 先转为 CPU 列表，device / dtype 转为字符串，JSON 不直接序列化 Torch 对象。model_manifest 的 llm 条目保存完整 cfg，输出记录中的 revision 与该条目一致。

#### 第五步：填写可重跑的模型记录

model_manifest.json 包含 model_id、实际 revision、模型类、tokenizer、代码版本、依赖、device 与 dtype。每个输出还保存 prompt/messages、生成参数、耗时和原始回答。

五个 prompt 固定后再运行，可覆盖概念解释、简单运算、格式输出等入门任务；样例结果用于了解本模型的行为。

加载计时只包含模型/tokenizer 加载阶段，结束前同步 GPU；加载后用一条 prompt 暖机，暖机不计入五条正式结果。每道正式题在 generate 前同步、reset_peak_memory_stats、开始 perf_counter，generate 后同步再结束计时；记录生成秒数、新增 token 数及 max_memory_allocated 的 MiB，解码时间单列或注明是否包含。显存是本进程 Torch 分配峰值，包含已加载模型。LLM/VLM 沿用同一字段名，保留实际环境和精度。

在本周 `experiments/vlm_qa/week8/README.md` 增加 LLM 张量流说明，用实际一题的 B、T、V 写出 `[B,T] → embedding → causal blocks → [B,T,V]`，联系第六周 MiniGPT；模型信息更新已有 model_manifest.json 的 llm 条目。

### 运行与验收

在云端激活 `petvision-vlm` 后执行：

```bash
python scripts/infer_llm.py --config configs/llm/llm_inference.yaml
```

`llm_outputs.jsonl` 应保留五条固定 prompt 的完整回答和设置。若回答重复了题目，先查是否只解码新增 ID；若遇到 tokenizer、模型类或 config 不兼容，先核对依赖版本与 revision，按对应版本模型卡修正，不通过更换题目掩盖问题。

- [ ] 云端环境与本地分开，GPU 和依赖有真实记录。
- [ ] Chat Template 与生成前缀正确，回答未把输入 prompt 一起重复当输出。
- [ ] 推理脚本可按固定配置和 revision 重跑。

<a id="day-3"></a>
## Day 3：真正的图像输入与 VLM 张量流

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [SmolVLM 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 图像+文本 messages、processor、最小推理 | 跑通一张真实图片 |
| [SmolVLM 官方文档](https://huggingface.co/docs/transformers/model_doc/smolvlm) | 视觉模块、连接与语言生成接口 | 查看实际 config 与模块 |
| [nn.Module hook（中文译文）](https://pytorch.ac.cn/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) | named_modules、register_forward_hook、register_forward_pre_hook、handle.remove | 临时记录实际模块形状 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 阅读所选模型实现与 processor |
| 2h | 实现 VLM 单图 smoke 与形状记录 |
| 1h | 画结构并核对图像确实进入模型 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `configs/vlm/vlm_inference.yaml`、`scripts/infer_vlm.py`，按固定 revision 的模型卡使用 processor 与兼容模型类；读取本地真实图片和文本消息。

2. 先支持 `--image`、`--question`、`--smoke` 的单图检查，打印输入字段、dtype、shape。input_ids 保持整数类型，浮点图像张量按模型精度处理；图片不能仅转换成分类标签传给纯文本 LLM。

3. 查看实际模块与 forward 路径，记录 pixel_values、视觉特征、连接/压缩后的图像 token、与文本组合后的序列及 logits 的形状；可用临时 hook，检查完移除。

4. 创建 `tensor_flow.md`，按实际模型画 Vision Encoder→连接/压缩模块→图像 token 与文本 embedding→Language Model。图像裁剪/多视图与特殊 token 数按 processor 输出记录，不硬套通用固定形状。

### 分步操作与自查

<a id="vlm-image-inputs"></a>
#### 第一步：加载 VLM，使用真实图片和配套 processor

创建 `scripts/infer_vlm.py`，顶层导入依赖，main 先解析 `--config`、`--image`、`--question`、`--smoke`，末尾添加入口保护；单图模式参数关系按第四步校验，Day 4 再添加批量 `--questions` 路径。下面的配置读取、模型加载、输入处理和生成都在这个脚本中接通。

先取一张能人工查看的本地图片，按模型卡构造含 image 与 text 的消息。processor 同时生成文本和图像相关字段，打印 key、dtype、shape。

将用于检查的图片准备为 `datasets/raw/vqa/example.jpg`，用 Pillow 打开并转为 RGB。该图可以作为单图开发样例，正式 Day 4 的评估图片和问题另行提前固定。创建 `configs/vlm/vlm_inference.yaml`：

```yaml
model:
  id: HuggingFaceTB/SmolVLM-256M-Instruct
  revision: null
data:
  batch_size: 1
  images_manifest: experiments/vlm_qa/week8/images_manifest.csv
inference:
  device: cuda
  dtype: float32
  attention_implementation: eager
  do_sample: false
  max_new_tokens: 128
output_file: experiments/vlm_qa/week8/vlm_outputs.jsonl
```

同样先查询本模型 SHA 并填回 revision。先用兼容当前模型的 Transformers 版本创建模型与 processor；下面采用支持该模型的 AutoModelForImageTextToText 接口，256M 模型的 Idefics3 架构接口可对照 [Transformers 官方说明](https://huggingface.co/docs/transformers/model_doc/idefics3)。模型卡中旧版 AutoModelForVision2Seq 示例需要配套库版本，不混用。把实际模型类和库版本写入 manifest。

在 infer_vlm.py 顶层导入 torch、PIL.Image、AutoProcessor、AutoModelForImageTextToText；main 读出 cfg 后加载，device / dtype 与昨天相同：

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model_id = cfg["model"]["id"]
revision = cfg["model"]["revision"]
if not isinstance(revision, str) or not revision:
    raise ValueError("请先把模型完整 commit SHA 填入 revision")
device = torch.device(cfg["inference"]["device"])
dtype = {"float32": torch.float32, "float16": torch.float16,
         "bfloat16": torch.bfloat16}[cfg["inference"]["dtype"]]
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, revision=revision,
    attn_implementation=cfg["inference"]["attention_implementation"],
).to(device=device, dtype=dtype).eval()
```

配置字段 attention_implementation 转为加载接口的 attn_implementation；首轮 eager，后续改精度或 Attention 实现时记录实际值。

messages 中要同时包含图像位置和问题；以模型卡中“单独传入本地图片”的接口为例，消息结构可先写为：

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "What objects are visible?"},
    ],
}]
```

单图分支把上面示例中的问题文本换成 args.question；图片占位符与实际 PIL 图片按下面这条完整路径处理：

```python
image = Image.open(args.image).convert("RGB")
messages[0]["content"][1]["text"] = args.question
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
for name, value in inputs.items():
    if isinstance(value, torch.Tensor):
        inputs[name] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        print(name, inputs[name].dtype, tuple(inputs[name].shape))
with torch.no_grad():
    generated = model.generate(
        **inputs, do_sample=cfg["inference"]["do_sample"],
        max_new_tokens=cfg["inference"]["max_new_tokens"],
    )
sequences = generated if isinstance(generated, torch.Tensor) else generated.sequences
new_ids = sequences[:, inputs["input_ids"].shape[-1]:]
answer = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
print(answer)
```

若所选兼容版本改用在 messages 中携带图片的接口，应整条替换为该版本模型卡的路径，不能省略图片预处理或混合两套占位符格式。

本节 AutoModelForImageTextToText 与单独 images 输入的组合可对照 [Idefics3 官方接口示例](https://huggingface.co/docs/transformers/model_doc/idefics3#transformers.Idefics3ForConditionalGeneration)。模型卡正文与不同版本文档可能展示其他 Auto 类；进入本周时同时核对已安装 Transformers 版本和实际模型 config，不把不同版本的导入、模板与 processor 调用拼成一段。这里提供待实现示例，是否通过以固定版本上的单图 smoke 为准。

```text
原图 → 尺寸调整/切图 → pixel_values
问题 → 模板/tokenizer → input_ids、attention_mask
上述字段共同输入 VLM
```

图片字段存在还不够，需要确认该字段真的传入实际模型 forward；不能在构建 messages 后遗漏图像处理结果。

迁移输入时保留 `input_ids` 为 long、布尔 mask 为布尔类型，只将需要的浮点图像张量转换成所选模型精度，保留 processor 返回的其他字段及其原有类型。若把整个输入字典都强制转为浮点数，embedding 查表会报错。先逐字段打印，再在单图上检查生成和新增 token 解码。

#### 第二步：定位视觉特征进入文本模型的位置

打印模型 config 和主要子模块，在实际视觉编码器输出、连接/压缩模块输出、进入语言模型的 embedding 处做一次小样本形状记录。

不同版本可能有多视图、裁剪或 token 压缩，不预先假定 pixel_values 必为四维或固定图像 token 数。以当前 processor 和模型实际结果为准。

先用 `model.named_modules()` 找视觉编码器、连接模块和语言模型的实际名称，再根据输出类型选 Tensor：输出若是 tuple 或 ModelOutput，先查看实际字段，再取对应的特征。以下是某个输出已确定为 Tensor 的模块的最小 hook 用法，target_module 为刚找到的真实模块：

```python
def show_output_shape(module, args, output):
    print(tuple(output.shape))

handle = target_module.register_forward_hook(show_output_shape)
try:
    with torch.no_grad():
        model(**inputs)
finally:
    handle.remove()
```

分别在视觉编码器和连接模块记录一次前向，不把完整特征 Tensor 长期存进列表。若进入语言模型的 embedding 通过 `inputs_embeds=` 关键字传入，使用带 `with_kwargs=True` 的 forward_pre_hook，在 `(module,args,kwargs)` 中查看 `kwargs["inputs_embeds"].shape`；普通位置参数 hook 看不到这些关键字。hook 只观察并返回 None，检查完成后移除所有 handle。

#### 第三步：建立自己的张量流表

创建 `experiments/vlm_qa/week8/tensor_flow.md`，至少列阶段、实际模块名、输入 shape、输出 shape、含义。图像特征和文本 embedding 汇合后，指出语言 decoder 如何据此生成回答；在本周 README 添加该文件的相对链接。

表中逐项填写 processor 图像字段、视觉编码器输出、连接/压缩后特征、文本 embedding、合并后语言输入和 logits；写出观察来自哪个模块/字段。某个合并发生在普通函数内时，查对应 forward 源码补一次临时 shape 打印，记录位置后移除。最终表使用本次真实图片和固定 revision 的结果，不能仅把通用结构图当作已观察的形状。

#### 第四步：smoke 只检查一张图

使用一条简单问题核对图像读入、输入设备、生成和解码。检查完移除临时 hook，再运行正式评估；正式原始回答由 Day 4 的固定问题集产生。

入口区分两种模式：`--image` 与 `--question` 配套用于单图，`--questions` 用于 Day 4 清单；两种输入方式互斥。`--smoke` 只打印该图的输入形状、回答和资源信息，不写或覆盖正式 `vlm_outputs.jsonl`。今天还没有 images manifest 时，单图分支直接使用命令中的图片路径。

将图片加载、messages 构造、processor、设备迁移、generate 和新增 ID 解码封装为文件顶层的 `infer_one(model, processor, image_path, question, device, dtype, generation_config)`，返回 raw_answer、新增 ID 列表和本题测量结果；main 负责加载模型一次，并把 args.image / args.question 传给它。Day 4 循环也调用这个函数，不能让内部仍读取单图模式的 args.image。与 LLM 一样校验 batch_size==1，单图模式要求同时提供 image、question 和 smoke；批量模式禁止 smoke，避免参数被静默忽略。单图通过后更新 model_manifest 的 vlm 条目，记录完整 cfg、模型类、processor 和实际依赖。

### 运行与验收

在云端已激活的 `petvision-vlm` 中执行：

```bash
python scripts/infer_vlm.py --config configs/vlm/vlm_inference.yaml --image datasets/raw/vqa/example.jpg --question 'What objects are visible?' --smoke
```

应看到真实图像相关字段、整数文本 ID、实际设备/精度和新增回答。图像 token 数不匹配时，先查 processor、模型 revision、图像占位符与图片数量；输入 dtype 报错时逐字段核对。OOM 时先检查是否同时保留了昨天的模型对象，再用 batch=1 和模型支持的图像尺寸定位。

- [ ] 模型输入包含真实图像 Tensor，单图回答和形状正常。
- [ ] 知道图像信息在哪里进入语言模型。
- [ ] 模块名、模型类、revision、输入尺寸与实际形状有记录。

<a id="day-4"></a>
## Day 4：20 张图、40 个问题的完整评估

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [所选 VLM 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 输入方式与已知能力边界 | 固定正式推理设置 |
| [总计划 VLM 验收](StudyPlan.md#stage-4) | 图片数、问题、原始输出和判定 | 准备完整样本记录 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 选 20+ 图片，先写问题与参考标准 |
| 1h | 批量推理并检查覆盖率 |
| 1.5h | 人工逐条判定和标记错误类型 |
| 0.5h | 核对记录完整性 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `images_manifest.csv`，记录至少 20 张可查看图片的 image_id、local_path、来源与 SHA256；图片放 datasets/raw/vqa。每张准备至少 2 题，在 questions.jsonl 写 qid、image_id、question、reference/criterion、question_type。

2. 题目整体覆盖类别、颜色、数量、位置和无法从图片判断的内容。参考答案或允许答案范围在推理前写好；信息不足题的正确行为可定义为明确说明无法判断。

3. 让 infer_vlm 支持 `--questions`，逐题独立构造 messages，写 vlm_outputs.jsonl：qid、原始回答、model/revision、生成设置、耗时、峰值显存与状态；读取或推理抛出的错误也保留记录。

4. 创建 judgments.csv，由人工按预先标准标 correct/partial/wrong，并记录错误类型和简短理由。核对每张至少 2 条、总数至少 40，统计正确数时报告总题数与失败数，不只保留成功回答。

### 分步操作与自查

#### 第一步：先建立两个可对应的清单

在 `experiments/vlm_qa/week8/` 下创建 `images_manifest.csv` 和 `questions.jsonl`；真实图片放在项目根的 `datasets/raw/vqa/`，所需目录先创建。清单可以在编辑器填写，批量读取与完整性校验接入 `scripts/infer_vlm.py` 的批量分支。

images_manifest.csv 每图一行，记录 image_id、local_path、source、sha256。local_path 统一相对于项目根目录，例如 datasets/raw/vqa/img001.jpg；在云端也放到同样的相对位置。图片自行选择可人工查看的样本并记录实际来源，CSV 只保存元数据，不包含图片内容。questions.jsonl 每题一行，通过 image_id 找图，通过 qid 唯一标识问题。

下面是记录格式示例，reference 需要看实际图片后提前填写：

```json
{"qid":"img001_q1","image_id":"img001","question":"What color is the cup?","reference":"按实际图片填写","question_type":"color"}
```

至少 20 个唯一 image_id，每个至少两题。题目整体覆盖类别、颜色、数量、位置和信息不足；不同问法也要有可人工核查的标准。

先用 csv.DictReader 读取图片清单、逐行 json.loads 读取问题；断言 image_id/qid 各自唯一，所有问题的 image_id 都在图片清单中，每张图片至少对应两题。逐图确认 local_path 可打开，并用 `hashlib.sha256(Path(local_path).read_bytes()).hexdigest()` 核对图片 bytes。标准可记录允许答案与部分正确条件，推理前固定并保存；正式生成后再判定。

#### 第二步：一次生成保留全部原始结果

修改已有 `scripts/infer_vlm.py`，添加 `--questions`（Path），读取该问题文件及 cfg["data"]["images_manifest"]。逐题通过 image_id 取得图片路径，复用 Day 3 已通过检查的单图推理函数；按 Day 3 第四步的互斥规则区分单图和批量模式。正式输出使用 cfg["output_file"]，本题为 `experiments/vlm_qa/week8/vlm_outputs.jsonl`，按第六周 JSONL 方法创建父目录并追加每题记录。

每题独立构造上下文，不把前一道题的答案喂给下一道。结果记录 qid、image_id、raw_answer、模型版本、设置、耗时、峰值显存和 status。

某题读取/推理失败也写一条 error 结果，保留原因并继续检查剩余样本；不要删除失败行来提高看上去的成功率。

循环中每题先建立含 run_id、qid、image_id 的记录，再在 try 中读取图片和生成；except 中写 status=error、raw_answer=None（序列化为 JSON null）和实际错误信息，最后将成功或失败记录统一追加一次。完整运行后，以当前 run_id 筛选记录检查每个 qid 恰好一条；重跑保留新 run_id 或单独输出文件，统计时明确选择一轮。

run_id 沿用 Day 2 的 uuid4 写法，在批量循环开始前生成一次。传给 infer_one 的 question 只取问题文本；reference / criterion 留给人工判定，不拼入模型输入。记录同时保留问题文本、图片 hash、实际模型 revision、生成参数、新增 IDs 和资源字段，使回答能对应到本次输入。成功状态写 ok，失败状态写 error；本步骤的 try/except 处理实际抛出的错误，尚未实现的强制超时不要写成已有能力。

#### 第三步：按预先标准人工判定

创建 `experiments/vlm_qa/week8/judgments.csv`，写入下面的表头与所选 run_id 的全部人工判定。评分由你查看真实图片、预先标准和原始回答后填写，不让生成模型直接给自身输出评分。

judgments.csv 为每个 qid 标 correct/partial/wrong 及错误类型和理由。颜色、数量等可明确核对；开放回答按提前写好的允许范围判定。信息不足题可要求明确承认无法确定。

建议列为 `run_id,qid,judgment,error_type,reason`，run_id 对应本轮选定的原始输出；错误类型可包括类别、属性、计数、空间关系、无依据回答与推理失败。读取/推理失败也保留一行，按本次整体任务统计为 wrong，并用错误类型区分。整体正确比例为 correct/全部题数，partial 数量另列，不在事后临时决定给多少分。

#### 第四步：核对覆盖率后再统计

问题清单的 qid 唯一；先用选定 run_id 筛选输出与判定，再检查这一轮的 qid 集合应一致且没有重复。报告总题数、成功生成数、失败数、正确数和部分正确数；分母清楚，结论限定在本次样本。

用一个手算例子检查统计：40 题中成功生成 38、失败 2，人工判定 correct=30、partial=5、wrong=5（含 2 条推理失败），严格正确比例为 `30/40=0.75`，部分正确另报 5 条；不能改用 38 作分母。统计前分别检查三份记录的 qid 唯一性，再比较集合相等，避免重复记录被集合去重掩盖。

### 运行与验收

确认图片、题目和判定标准均已固定后，在云端执行：

```bash
python scripts/infer_vlm.py --config configs/vlm/vlm_inference.yaml --questions experiments/vlm_qa/week8/questions.jsonl
```

核对唯一图片数至少 20、每图至少两题，questions、outputs、judgments 的 qid 集合完全一致。重新运行某题时保留运行标识与原结果，并明确本轮统计选择哪次输出；不通过重复追加同一个 qid 增加题数。结果下载回本地时保留配置、model manifest 与 cloud 环境快照。

- [ ] 图像、问题、原始回答与人工判定均能通过 ID 对应。
- [ ] 至少 20 张图片、40 条完整记录，包含失败结果。
- [ ] 评估标准提前确定，结论限定在本次样本与模型配置。

<a id="day-5"></a>
## Day 5：跨模块复盘与最终报告初稿

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [总计划验收清单](StudyPlan.md#acceptance) | 基础、视觉、语言、多模态与复现要求 | 逐项核对证据 |
| [本项目总计划目录](StudyPlan.md#layout) | 模块与报告分工 | 整理已有实验入口 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 核查八周成果与缺口 |
| 2h | 撰写最终报告并链接实际结果 |
| 1h | 完成模型结构与失败分析 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 创建 `docs/final_report.md`，按基础训练、视觉对照、语言建模、CLIP/LLM/VLM 组织，链接各周 README 和真实结果，不重复复制全部日志。

2. 视觉部分说明固定划分、三类对照与最终测试；语言部分说明同 tokenizer 与跨 tokenizer 的指标边界；VLM 部分列样本范围、判断标准、失败类型和资源开销。

3. 补全 LLM/VLM 模型 manifest 与 tensor_flow，记录本地/云端环境、权重 revision、运行命令、精度和耗时方法。

4. 对总计划逐项标记已完成、待补、拓展；无法证明的能力如实保留，不把运行过预训练模型等同于完成微调。

### 分步操作与自查

#### 第一步：按问题组织最终报告

创建 `docs/final_report.md`，将下面的结构作为该文件的章节。链接从 docs/ 计算：各周说明用 `../experiments/<主题>/weekN/README.md`，周指南用 `weekN.md`；待文件实际产生后再添加结果链接，不用示例数值冒充实验结论。

建议使用以下结构，每节引用真实结果而非复制全部日志：

```text
1. 学习目标、预算与实际环境
2. 基础：梯度、形状与训练流程
3. 视觉：BN、残差、Patch 对照与测试
4. 语言：字符基线、BPE、采样与续训
5. 多模态：CLIP、LLM、VLM 和 40+ 题判定
6. 失败分析、局限、待补事项与下一轮优先级
```

#### 第二步：给每个结论附可检查证据

准确率链接对应测试表和 checkpoint 标识；语言 loss 写 tokenizer、split 和有效 token 口径；VLM 指标链接原始输出与人工判定。

只有少量样例时表述为本次观察，不扩大为所有数据、模型或任务上的结论。

逐项核对「结论 → 本次设置 → 真实证据 → 局限」：例如视觉结论链接六变体测试表与三类对照曲线；语言结论附 tokenizer_hash、split 和目标数；VLM 结论附当前 run_id 的全部输出、判定及上述统计分母。结论尚无证据时列为待补，报告中的示例数字不替代实际运行值。

#### 第三步：明确三种训练来源

主线 CNN/ResNet/ViT/Mini-GPT 是自己从零训练；CLIP/LLM/VLM 使用预训练权重推理；微调属于后续路线。报告中记录各自初始化和权重来源。

#### 第四步：如实保留缺口

对照总计划列已完成、待补和拓展。运行过脚本、检查通过、能够解释、能够有限参考重写分别对应不同证据，不能互相替代。

继续更新 `experiments/vlm_qa/week8/model_manifest.json` 和 `experiments/vlm_qa/week8/tensor_flow.md`，核对 CLIP/LLM/VLM 的实际 SHA、模型类、精度、运行命令、两端环境目录和测时范围，并从 final_report.md 链接这两份文件。总计划中的验收勾选只对应已有证据，未通过项保留待补。

### 运行与验收

今天根据已有实验记录整理报告，不新增训练任务。逐项核对 `docs/final_report.md`、`model_manifest.json` 和 `tensor_flow.md`：报告链接能打开，指标注明 split/tokenizer/判定口径，模型条目能找到对应实际环境和配置。缺少结果的项明确保留为待补。

- [ ] 报告每个重要结论有具体结果或代码入口支撑。
- [ ] 模型与环境版本、数据来源、判定标准和失败结果齐全。
- [ ] 未完成事项与八周之后的拓展范围清楚。

<a id="day-6"></a>
## Day 6：复现抽查、解释与最终整理

### 去哪里学、读到哪里

| 资料 | 今天只读这些内容 | 阅读后立刻做 |
| --- | --- | --- |
| [总计划六个问题](StudyPlan.md#routine) | 梯度、训练、视觉、Attention、GPT 与 VLM | 检查能否自己解释 |
| [各周 README](../README.md#文档导航) | 选择一项代表实验回看复现说明 | 修复导航与命令 |

### 时间与操作

| 时间 | 做什么 |
| --- | --- |
| 1h | 抽查代表性脚本、checkpoint 和链接 |
| 1.5h | 解释核心模块与实际失败案例 |
| 1h | 修订最终报告和后续优先级 |
| 0.5h | 完成本周 README 与文档导航 |
| 1h 机动 | 补基础、定位报错、核对形状与复盘当天卡点 |

### 具体怎么做

1. 从 NumPy 梯度、分类加载、因果 Mask、tokenizer 往返、VLM 单图中各选一个小检查；按现有说明抽查，避免为了收尾重跑全部大实验。

2. 口头或在报告中解释六个核心问题，并选择一个误分类、一个生成问题和一个视觉问答失败，说明证据与下一步。

3. 更新本周 README 的实际命令、模型版本、指标和结果链接；根 README 继续专注介绍与文档导航，首次操作、排错和验收留在逐日指南，实际复现索引留在各周 README。

4. 把基础关卡未通过项排到下一轮前面，再安排 Encoder–Decoder、迁移学习、LoRA/QLoRA、VLM 微调或 App；依据实际兴趣和时间选择，不将其补进本轮必做。

### 分步操作与自查

#### 第一步：按说明抽查最小复现

每个阶段选一个小入口：NumPy 梯度、分类加载、causal Mask、tokenizer 往返、VLM 单图。确认文档中的路径和参数仍与实际文件一致。

抽查以证明材料可用为目标；已有大实验结果保留，只有发现实现错误或配置缺失时才重跑相关部分。

#### 第二步：解释六个核心问题

说明手写梯度与 autograd、训练/验证/测试、CNN 与残差、QKV/Mask/CLS、GPT 与 BPE，以及图像进入 VLM 的方式。尽量结合自己的具体 shape 和一次错误。

#### 第三步：整理导航和结果位置

最终报告链接各周 README；每周 README 链接对应配置与结果。根 README 继续用于项目介绍与导航，逐日指南保留完整学习步骤和当天命令，各周 README 保存实际复现索引。

#### 第四步：形成下一轮可执行清单

把仍未通过的基础关卡放在前面，再选择拓展方向。每项写清学习问题、最小实验和验收方式，避免只罗列更多模型名称。

### 运行与验收

在本地项目根目录的 `petvision` 中抽查基础、分类加载和核心模块：

```bash
python scripts/gradient_check.py
python scripts/evaluate.py --config configs/cv/mlp_mnist.yaml --checkpoint outputs/mnist_mlp/best.pt --split val
python -m pytest tests/test_attention.py tests/test_tokenizer.py -q
```

再在云端 `petvision-vlm` 中重跑 Day 3 的单图 `--smoke` 命令，确认真实图片仍能进入模型；该分支不覆盖 Day 4 正式结果。已有大实验无需全部重跑，只对抽查发现的问题补充修复和验证，随后检查最终报告与结果索引。

- [ ] 最小复现抽查通过，文档与真实目录一致。
- [ ] 完整保存 VQA 40+ 条结果、判定和失败分析。
- [ ] 最终报告能说明已完成的能力与仍需巩固的部分。

## 本周结束检查

- [ ] CLIP、LLM、VLM 各有可重跑入口与模型/环境记录。
- [ ] VLM 输入真实图片，结构与关键形状按实际实现说明。
- [ ] 20+ 图片、40+ 问答有预先标准、全部原始输出和人工判定。
- [ ] 最终报告链接各周真实结果，未完成项与拓展如实区分。

先用本周 6 小时机动补齐未完成项；仍未通过时，下一周先处理该关卡，后续任务顺延。周末小结控制在约 10–20 分钟，记录实际配置、关键结果、一个解释和仍需巩固的部分。

主线结束后，按总计划整理未完成事项，再决定八周之后的拓展路线。

