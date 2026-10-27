# 第八周：CLIP、LLM / VLM 推理与八周总结

学习指南：[第八周逐日任务](../../../docs/week8.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

本地 CLIP 的完整依赖、配置和版本固定步骤见[Day 1](../../../docs/week8.md#clip-environment)，云端环境创建、安装与排错见[Day 2](../../../docs/week8.md#cloud-environment)。实际使用两套环境：

| 位置与用途 | Conda 环境 | 环境记录目录 |
| --- | --- | --- |
| 本地 CLIP | `petvision` | `environments/local/` |
| 云端 LLM / VLM | `petvision-vlm` | `environments/cloud/` |

各目录保存 environment.txt、conda-packages.txt、conda-explicit.txt 和 pip-freeze.txt。实际安装命令与版本在最小推理验证后填写，云端定义同步到根 `environment-vlm.yml`；尚未验证的组合保留待填写。两端快照分别保存，避免结果同步时相互覆盖。

model_manifest.json 记录每个模型 ID、完整 commit SHA、模型类、processor/tokenizer、精度、device、代码版本及所用环境目录。模型和 processor 使用相同 revision；首轮配置中的 null 在正式推理前替换成已核验 SHA。云端结果同步回本地本周目录，模型权重留在缓存/outputs。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1，本地 petvision | `python scripts/run_clip.py --config configs/cv/clip_zeroshot.yaml` | 固定验证子集零样本分类 |
| Day 2，云端 petvision-vlm | `python scripts/infer_llm.py --config configs/llm/llm_inference.yaml` | 固定五个 prompt 的文本推理 |
| Day 3，云端 petvision-vlm | `python scripts/infer_vlm.py --config configs/vlm/vlm_inference.yaml --image datasets/raw/vqa/example.jpg --question 'What objects are visible?' --smoke` | 真实图片与形状检查 |
| Day 4，云端 petvision-vlm | `python scripts/infer_vlm.py --config configs/vlm/vlm_inference.yaml --questions experiments/vlm_qa/week8/questions.jsonl` | 完整 40+ 题，保留全部原始输出 |

## 配置与结果

CLIP 做候选图文匹配，LLM 做纯文本推理，VLM 接收真实图像参与推理，VQA 是视觉问答任务。processor、revision、greedy 和各类 ID 的含义见[本周术语表](../../../docs/week8.md#inference-terms)。40+ 题表示至少 40 道题；这周命令使用预训练模型进行推理，不包含模型微调。

入门模型：CLIP `openai/clip-vit-base-patch32`；LLM `HuggingFaceTB/SmolLM2-360M-Instruct`；VLM `HuggingFaceTB/SmolVLM-256M-Instruct`。按当前兼容模型卡先验证，再固定实际 revision 与依赖。LLM/VLM batch=1、do_sample=False、max_new_tokens=128；记录精度、上下文、图像处理、生成耗时与峰值显存。CLIP 使用模型 processor 和固定英文模板，零样本结果单列。VQA 原始样本和标准在正式推理前固定，统计含全部失败记录。所选 SmolVLM 模型卡标注主要语言为英语，本轮优先英文问答；如增加中文题，单独记录语言条件并解释结果。

LLM / VLM 当前按 batch=1 逐题调用，其他 batch 值需明确拒绝。每轮生成一次 run_id，原始输出与判定使用同一个运行标识；VLM 单图和批量模式复用 infer_one，参考答案只用于人工判定，不进入模型输入。模型 manifest 保留每次使用的完整配置及实际兼容版本。

CLIP 创建和图文输入见[Day 1](../../../docs/week8.md#clip-inputs)，LLM 的配置、模型 / tokenizer 加载及生成见[Day 2](../../../docs/week8.md#llm-inference)，VLM 的完整图片输入路径见[Day 3](../../../docs/week8.md#vlm-image-inputs)。judgments.csv 带 run_id，与本轮选择的原始输出对应；统计前先筛选同一轮。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `clip_subset.json / clip_results.csv` | Day 1：固定图像索引、类别分数、预测与标签 |
| `figures/clip_mistakes.png` | Day 1：有错分时保存最多 8 张原图及真实 / 预测类 |
| `model_manifest.json / environments/{local,cloud}/` | 模型版本、分别保存的两端环境及安装信息 |
| `llm_outputs.jsonl` | 五个固定 messages 的原始输出和配置 |
| `images_manifest.csv / questions.jsonl` | 20+ 图片与预先写好的 40+ 题标准 |
| `vlm_outputs.jsonl / judgments.csv` | 全部原始回答、状态与人工判定 |
| `tensor_flow.md` | 实际视觉模块、连接、token 与解码形状 |
| `docs/final_report.md` | 项目根下：八周报告、证据链接与后续清单 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

