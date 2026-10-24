# 第五周：小型 Vision Transformer 与视觉模型比较

学习指南：[第五周逐日任务](../../../docs/week5.md)  
总计划：[StudyPlan.md](../../../docs/StudyPlan.md)

本文件保存实际环境、复现命令索引、结果链接和周总结。首次学习时按上方逐日指南中的安装、操作、运行与排错步骤顺序完成；这里便于之后查找和重跑。脚本、配置与结果在学习中逐步建立，实际指标在运行后填写。

<a id="environment"></a>
## 环境准备

沿用 Conda `petvision`、第二周安装的 torch / torchvision、PyYAML、pytest。依赖没有变化时复用此前版本记录；改变精度、batch、设备需同步记入两组配置。

```bash
conda activate petvision
python -m pip check
```

从零训练，不下载预训练 ViT 权重。实际显存不足时统一减小同一对实验的 batch，保持比较口径。

<a id="commands"></a>
## 运行练习

以下命令均从项目根目录执行，先激活对应 Conda 环境。参数是本周要实现的接口约定。形状与数值检查直接打印结果；环境记录、实验指标、生成样例和模型输出按下表保存。

| 学习日 | 运行入口 | 用途 |
| --- | --- | --- |
| Day 1–2 | `python -m pytest tests/test_vit.py -q` | Patch、CLS、位置、前反向检查 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --smoke` | ViT 短运行 |
| Day 2 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml --overfit-samples 64` | 小样本拟合 |
| Day 3 | `python scripts/train_cv.py --config configs/cv/vit_cifar10.yaml` | Patch=4 |
| Day 3 | `python scripts/train_cv.py --config configs/cv/vit_patch8_cifar10.yaml` | Patch=8 |
| Day 4 | `python scripts/evaluate.py --config configs/cv/vit_cifar10.yaml --checkpoint outputs/vit_patch4/best.pt --split val` | Patch=4 验证与统一指标 |
| Day 4 | `python scripts/evaluate.py --config configs/cv/vit_patch8_cifar10.yaml --checkpoint outputs/vit_patch8/best.pt --split val` | Patch=8 验证与统一指标 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/cnn_cifar10.yaml --checkpoint outputs/cnn_bn_on/best.pt --split test` | CNN BN on 测试 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/cnn_no_bn_cifar10.yaml --checkpoint outputs/cnn_bn_off/best.pt --split test` | CNN BN off 测试 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/resnet_cifar10.yaml --checkpoint outputs/resnet_on/best.pt --split test` | 残差 on 测试 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/resnet_no_residual_cifar10.yaml --checkpoint outputs/resnet_off/best.pt --split test` | 残差 off 测试 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/vit_cifar10.yaml --checkpoint outputs/vit_patch4/best.pt --split test` | Patch=4 测试 |
| Day 5 | `python scripts/evaluate.py --config configs/cv/vit_patch8_cifar10.yaml --checkpoint outputs/vit_patch8/best.pt --split test` | Patch=8 测试 |

## 配置与结果

小 ViT：C=128、depth=2、heads=4、FFN=512、dropout=0.1、AdamW lr=3e-4、weight_decay=0.01、20 epochs、batch=64、seed=42。两组 token 长度分别为 65/17。固定复用第三周 CIFAR 索引；各组 experiment_dir 使用 `experiments/vit_patch_size/week5/runs/patch4` 或 `patch8`，checkpoint 位于 `outputs/vit_patch4` / `outputs/vit_patch8`。若短运行测时后统一调整预算，保留实际配置。best 按验证 accuracy 选择，持平保留较早一次，last 为最后一轮。两组记录独立初始化还是复制公共层；复制时显式排除形状不同的 Patch/位置参数，不能直接套用第三周对所有共有键形状一致的加载规则。随机增强与 dropout 的实现和单种子局限见逐日指南。

复用边界见[Day 2 模块组合](../../../docs/week5.md#encoder-reuse)：ViT 复用第四周 EncoderBlock 的类，创建两个新的 Block；CLS、可学习位置和堆叠后的最终 LayerNorm 由 ViT 管理。forward 中位置相加一次、最终 LayerNorm 一次，Block 内部 norm1/norm2 保留；Attention 为双向。

统一指标接入 evaluate.py 的步骤见[Day 4](../../../docs/week5.md#classification-metrics)；冻结清单写在本 README，列出六个变体实际的配置、best 路径和验证结果。

| 文件或目录 | 内容与产生时间 |
| --- | --- |
| `patch_results.csv` | Day 4：两组 best 的验证结果、token 数、参数量与时间 |
| `visual_test_results.csv` | Day 5：六个冻结变体的 test 指标 |
| `figures/patch_comparison.png` | 两种 Patch 的训练/验证曲线 |
| 各组 `experiment_dir/figures/confusion_{val,test}.png`、`mistakes.png` | 各变体混淆矩阵与真实误分类，按实际运行目录索引 |
| `runs/patch4 / runs/patch8` | 各组配置快照和 history |
| `outputs/vit_patch4 / outputs/vit_patch8` | 项目根下：best/last checkpoint |

结果产生后将路径改为可点击链接，并填写对应运行命令和配置；不提前填写指标。大权重与原始数据保存在忽略目录。

## 周总结

- 完成情况与仍需补齐的关卡：待本周结束后填写。
- 关键实验结果与结论：待实际运行后填写。
- 一两条关键理解与典型错误：待填写。
- 后续要巩固的部分：待填写。

保留已有曲线、结果表和配置快照，周末用约 10–20 分钟简要总结。日常验收直接检查代码、输出、图表与解释。

