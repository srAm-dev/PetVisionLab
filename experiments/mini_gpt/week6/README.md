# 第六周：字符级 Bigram 与 Mini-GPT

学习指南：[第六周逐日任务](../../../docs/week6.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 Conda `petvision` 与前几周依赖，不需要下载预训练 GPT 权重。数据来源见 [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)，下载内容和处理规则在 split manifest 记录。

```bash
conda activate petvision
python -m pip check
```

先按小模型配置测时与显存，若统一缩小 batch/context/steps，需写入两组配置与实验报告。训练与验证数据窗口来自各自原文文件。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1 | `python scripts/prepare_text.py` | 下载记录、80/10/10 原文划分与字符词表 |
| Day 2–4 | `python -m pytest tests/test_language_model.py -q` | 错位、形状、因果性与保存加载检查 |
| Day 4 | `python scripts/train_gpt.py --config configs/llm/bigram_char.yaml` | 接通保存后正式训练字符 Bigram |
| Day 4 | `python scripts/train_gpt.py --config configs/llm/mini_gpt.yaml` | 接通保存后正式训练字符 Mini-GPT |
| Day 5 | `python scripts/generate.py --checkpoint outputs/bigram_char/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --output experiments/mini_gpt/week6/samples.jsonl` | Bigram 生成 |
| Day 5 | `python scripts/generate.py --checkpoint outputs/mini_gpt/best.pt --prompt 'ROMEO:' --max-new-tokens 200 --seed 42 --output experiments/mini_gpt/week6/samples.jsonl` | GPT 生成 |

## 配置与结果

相同字符 tokenizer、原文划分、context=128、batch=16、seed=42、AdamW lr=3e-4、weight_decay=0.01、2000 steps、每 200 步验证。GPT 为 2 层/C128/4 heads/FFN512/dropout0.1；Bigram 为 V×V 转移 logits。各组 experiment_dir 为本周 `runs/bigram`、`runs/gpt`，checkpoint 位于项目根 `outputs/bigram_char`、`outputs/mini_gpt`。测试留到第七周；同 tokenizer 下才比较 token 级 loss/PPL。

从文本 / tokenizer 文件读取到 token IDs、sample_batch 和语言训练函数的接线见[Day 2](../../../docs/week6.md#text-inputs)；train_step / evaluate_lm 放在 src/petvision/training/lm_trainer.py。last 在验证时点和最终步保存，恢复以文件中的 step 为准。

两个训练函数统一返回 loss / n_target_tokens。即使最终 steps 不是 eval_interval 的整数倍，末步也执行完整验证并保存；每次只追加当前 row。各运行独立保存 figures/loss_curves.png，两组都完成后再生成周目录下的合并图，BPE 可复用同一入口。

完整 YAML 和创建步骤见[第六周指南](../../../docs/week6.md)。Day 2–3 先做小规模检查，Day 4 完成保存后才启动正式 2000 步；`--output` 将完整生成样例追加到指定 JSONL 文件。

验证使用[Day 2 固定分窗规则](../../../docs/week6.md#eval-windows)：长度 N 的 split 覆盖索引 1..N-1 的全部目标，窗口共享边界输入但不重复统计目标；尾窗按实际长度评估。总 NLL 除以 N-1，日志记录实际目标数；evaluate 结束后恢复调用前的 train/eval 模式，训练更新时启用 train；两组使用相同窗口和上下文重置方式，第七周最终测试沿用此规则。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `split_manifest.json` | 原文来源、SHA256、边界和规范化规则，第七周复用 |
| `char_tokenizer.json` | 训练集字符映射、UNK 及格式版本 |
| `runs/bigram / runs/gpt` | 各组 config、逐验证点追加的 history、run_summary.json 和单模型曲线 |
| `results.csv / figures/loss_curves.png` | 同 tokenizer 的验证结果及曲线 |
| `samples.jsonl` | 真实生成文本、prompt、seed 与模型信息 |
| `outputs/bigram_char / outputs/mini_gpt` | 项目根下：best/last 与配套模型信息 |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

