# 第五周：小型 Vision Transformer 与视觉模型比较

学习指南：[第五周逐日任务](../../../docs/week5.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

Day 1 先写 `vit.py` 和 Patch 检查，Day 2 接入分类入口。Day 3 训练完成后，从两个运行目录取配置快照和 best。Day 4 实现按 split 写指标的评估分支，生成两份 ViT 验证 JSON，再和第三周四组模型一起冻结。Day 5 才执行六条测试命令。[第三周交接表](../../cnn_vs_resnet/week3/README.md#week5-handoff)

<a id="environment"></a>

## 环境准备

沿用 Conda `petvision`、第二周安装的 torch / torchvision、PyYAML、pytest。依赖没有变化时复用此前版本记录；改变精度、batch、设备需同步记入两组配置。

```bash
conda activate petvision
python -m pip check
```

从零训练，不下载预训练 ViT 权重。实际显存不足时统一减小同一对实验的 batch，保持比较口径。

<a id="commands"></a>

## 实现后的运行命令

从项目根目录运行。先按[第 5 周指南](../../../docs/week5.md#day-1)准备当天的入口、配置和输入，再执行该行命令。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1-2 | `python -m pytest tests/test_vit.py -q` | Patch、CLS、位置、前反向检查 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --smoke` | ViT 短运行 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --overfit-samples 64` | 小样本拟合 |
| Day 3 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml` | Patch=4 |
| Day 3 | `python scripts/train_cv.py --config configs/cv/vit_patch8_cifar10.yaml` | Patch=8 |
| Day 4 | `python scripts/evaluate.py --config experiments/vit_patch_size/week5/runs/patch4/config.yaml --checkpoint outputs/vit_patch4/best.pt --split val` | Patch=4 验证与统一指标 |
| Day 4 | `python scripts/evaluate.py --config experiments/vit_patch_size/week5/runs/patch8/config.yaml --checkpoint outputs/vit_patch8/best.pt --split val` | Patch=8 验证与统一指标 |
| Day 5 | `python scripts/evaluate.py --config experiments/cnn_vs_resnet/week3/runs/cnn_bn_on/config.yaml --checkpoint outputs/cnn_bn_on/best.pt --split test` | CNN BN on 测试 |
| Day 5 | `python scripts/evaluate.py --config experiments/cnn_vs_resnet/week3/runs/cnn_bn_off/config.yaml --checkpoint outputs/cnn_bn_off/best.pt --split test` | CNN BN off 测试 |
| Day 5 | `python scripts/evaluate.py --config experiments/cnn_vs_resnet/week3/runs/resnet_on/config.yaml --checkpoint outputs/resnet_on/best.pt --split test` | 残差 on 测试 |
| Day 5 | `python scripts/evaluate.py --config experiments/cnn_vs_resnet/week3/runs/resnet_off/config.yaml --checkpoint outputs/resnet_off/best.pt --split test` | 残差 off 测试 |
| Day 5 | `python scripts/evaluate.py --config experiments/vit_patch_size/week5/runs/patch4/config.yaml --checkpoint outputs/vit_patch4/best.pt --split test` | Patch=4 测试 |
| Day 5 | `python scripts/evaluate.py --config experiments/vit_patch_size/week5/runs/patch8/config.yaml --checkpoint outputs/vit_patch8/best.pt --split test` | Patch=8 测试 |

## 计划配置与待保存结果

P 是 Patch 边长；32×32 图片切出 N=(32/P)² 个 Patch，再加一个 CLS 分类 token，所以 P=4/8 的序列长度为 65/17。C 是每个 token 的特征长度，不是 RGB 通道数；完整形状与参数解释见[本周符号表](../../../docs/week5.md#vit-shapes)，F1 统计见[Day4](../../../docs/week5.md#day-4)。

小 ViT：C=128、depth=2、heads=4、FFN=512、dropout=0.1、AdamW lr=3e-4、weight_decay=0.01、20 epochs、batch=64、seed=42。两组 token 长度分别为 65/17。固定复用第三周 CIFAR 索引；各组 experiment_dir 使用 `experiments/vit_patch_size/week5/runs/patch4` 或 `patch8`，checkpoint 位于 `outputs/vit_patch4` / `outputs/vit_patch8`。若短运行测时后统一调整预算，保留实际配置。best 按验证 accuracy 选择，持平保留较早一次，last 为最后一轮。首轮两组都用 seed=42 独立初始化，并记录这种设置。若另做公共层对齐实验，按指南排除形状不同的 Patch 与位置参数。随机增强与 dropout 的实现和单种子局限见逐日指南。

模型组合见[Day 2](../../../docs/week5.md#encoder-reuse)：在 `src/petvision/models/cv/vit.py` 定义 ViTMLP 和 ViTBlock，后者调用第四周的 MultiHeadSelfAttention 与 LayerNorm，使用 pre-LN＋GELU。ViT.forward 按 Patch→CLS/位置相加→embedding dropout→独立 ViTBlock 堆叠→取第 0 个 token→head(LayerNorm＋Linear) 计算；位置只加一次，Attention 为双向。

数据处理复用第三周的 [torchvision.transforms.v2 流程](../../../docs/week3.md#cifar-v2-transforms)：ToImage 保留 uint8 像素，训练时加入随机裁剪和翻转，再用 ToDtype(torch.float32,scale=True) 缩放、Normalize 归一化；验证和测试省略随机增强。各变体按冻结配置对应的确定性预处理评估：CNN / ViT 的模型输入为 32×32，ResNet 在 ToDtype 之前 Resize 到 96×96。

修改 `scripts/evaluate.py::main`，调用该脚本的 evaluate_classification 完整统计混淆矩阵、loss 与样本数，再调用 benchmark_forward 记录推理测量；F1 计算由 `src/petvision/evaluation/classification.py::metrics_from_confusion` 完成，细节见[Day 4](../../../docs/week5.md#classification-metrics)；冻结清单写在本 README，列出六个变体实际的配置、best 路径和验证结果。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `patch_results.csv` | Day 4：两组 best 的验证结果、token 数、参数量与时间 |
| `visual_test_results.csv` | Day 5：六个冻结变体的 test 指标 |
| `figures/patch_comparison.png` | 两种 Patch 的训练/验证曲线 |
| 各组 `experiment_dir/figures/confusion_{val,test}.png`、`mistakes_{val,test}.png` | 各变体混淆矩阵与真实误分类，按实际运行目录索引 |
| `runs/patch4 / runs/patch8` | 各组 config、history、run_summary.json、val/test_metrics.json 和图表 |
| `outputs/vit_patch4 / outputs/vit_patch8` | 各自的 initial、best 和 last checkpoint |

## 周总结

- Patch 对照：待填写 P4/P8 的验证结果、token 数、参数量与实测耗时。
- 视觉测试：待链接六个冻结变体的测试表，说明输入尺寸和预算差异。
- 错误与理解：待填写一个误分类分析、ViT 重写检查和后续要巩固的部分。
