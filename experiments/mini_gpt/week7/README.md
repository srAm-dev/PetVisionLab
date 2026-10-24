# 第七周：简化字节级 BPE、采样与中断续训

学习指南：[第七周逐日任务](../../../docs/week7.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 `petvision`，BPE 核心用 Python 标准库实现，不需要额外 tokenizer 库。已有 torch/pytest 足够完成检查，完整操作按[第七周指南](../../../docs/week7.md)进行。依赖若变化，记录实际命令与快照。

```bash
conda activate petvision
python -m pip check
```

训练原文与 `experiments/mini_gpt/week6/split_manifest.json` 复用第六周，不重新随机划分。BPE 字节词表和 merges 作为独立 JSON 配套保存，字符 tokenizer 保留。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1–2 | `python -m pytest tests/test_tokenizer.py -q` | 合并、保存加载、Unicode 往返 |
| Day 2 | `python scripts/train_tokenizer.py --vocab-size 512 --max-train-chars 100000` | 训练教学 BPE |
| Day 3 | `python -m pytest tests/test_sampling.py -q` | 采样分布与复现 |
| Day 3 | `python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 0.7 --top-k 20 --output experiments/mini_gpt/week7/samples.jsonl` | 较集中采样 |
| Day 3 | `python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 1.0 --output experiments/mini_gpt/week7/samples.jsonl` | 不启用 top-k 的对照 |
| Day 4 | `python -m pytest tests/test_resume.py -q` | 连续 40 与 20+20 恢复检查 |
| Day 5 | `python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml` | 创建配置后，从零训练 BPE GPT |
| Day 5，确实中断且已有未完成的 last 时 | `python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml --resume outputs/mini_gpt_bpe/last.pt` | 以相同配置继续到 training.steps，不额外训练一轮 |
| Day 5 | `python scripts/compare_tokenizers.py` | 相同 val 原文的编码长度和耗时 |
| Day 5 | `python scripts/generate.py --checkpoint outputs/mini_gpt_bpe/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --temperature 0.7 --top-k 20 --output experiments/mini_gpt/week7/samples.jsonl` | BPE 生成 |
| Day 6 | `python scripts/train_gpt.py --config configs/llm/bigram_char.yaml --evaluate-only --checkpoint outputs/bigram_char/best.pt --split test` | 字符 Bigram 最终测试 |
| Day 6 | `python scripts/train_gpt.py --config configs/llm/mini_gpt.yaml --evaluate-only --checkpoint outputs/mini_gpt/best.pt --split test` | 字符 GPT 最终测试 |
| Day 6 | `python scripts/train_gpt.py --config configs/llm/mini_gpt_bpe.yaml --evaluate-only --checkpoint outputs/mini_gpt_bpe/best.pt --split test` | BPE GPT 单列最终测试 |

## 配置与结果

BPE 起始 256 个字节，目标词表 512，训练子集只取 train 原文前 100,000 个字符，记录 hash、实际合并数与训练时间。BPE GPT 复用第六周结构/2000 步预算，从新词表对应的新模型训练；experiment_dir 为本周 `runs/bpe`，checkpoint 为 `outputs/mini_gpt_bpe`。续训检查单独使用 CPU 小模型 40 与 20+20 步，记录最大差异。采样组合比较始终使用同一 checkpoint 与 prompt；两组同时改变温度与 top-k，只描述组合效果。

Day 4 的 `test_resume.py` 自带小模型配置和教学 token 数据，在 pytest 临时目录保存/读取 checkpoint，独立于正式 BPE 配置和输出。Day 5 才创建 mini_gpt_bpe.yaml 并启动正式训练；按[执行顺序](../../../docs/week7.md#bpe-resume-order)，先从零跑，确实中断且 `0 < last.step < training.steps` 时才使用恢复命令。已达到累计目标则直接评估，不需要再次训练或人为中断。恢复时保留 best_val_loss/best_step、原 best 文件和训练统计累计值，history 只追加恢复点之后的验证行，避免覆盖更优结果或重复记录。

tokenizer 训练脚本的参数和调用顺序见[Day 2](../../../docs/week7.md#bpe-training-entry)；训练、生成与编码比较统一通过[load_tokenizer](../../../docs/week7.md#tokenizer-dispatch)按 JSON type 选择字符 / BPE 实例。

所有模型的最终测试沿用[第六周验证窗口规则](../../../docs/week6.md#eval-windows)：同一 split 内覆盖 N-1 个目标、保留尾窗、按目标 token 数汇总。字符与 BPE 的 N 由各自 tokenizer 决定，指标仍分开报告。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `bpe_tokenizer.json` | 有序 merges、bytes 词表、训练子集 hash |
| `tokenizer_comparison.csv` | 相同 val 原文的 token 数、字节数与编码耗时 |
| `samples.jsonl` | 真实采样文本、token IDs、模型和参数 |
| `runs/bpe/` | BPE GPT 配置、train/val history 与曲线 |
| `outputs/mini_gpt_bpe/` | 项目根下：模型、优化器、随机状态与 tokenizer 标识 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

