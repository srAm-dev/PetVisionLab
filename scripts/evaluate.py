import yaml
import json
import hashlib
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import evaluate
from petvision.utils.checkpoint import load_checkpoint
from petvision.data.datasets import build_mnist_loaders

def collect_mistakes(model, loader, device, limit=8):
    mistakes = []
    if limit <= 0:
        return mistakes

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)               # [B, 10]
            probabilities = logits.softmax(dim=1)
            predicted = logits.argmax(dim=1)     # [B]

            # 找到当前批次中预测错误的样本位置。
            wrong_indices = (
                (predicted != labels)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )

            for index in wrong_indices:
                predicted_label = predicted[index].item()

                mistakes.append({
                    "image": images[index].cpu(),
                    "true_label": labels[index].item(),
                    "predicted_label": predicted_label,
                    "score": probabilities[index, predicted_label].item(),
                })

                if len(mistakes) >= limit:
                    return mistakes

    return mistakes

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    return args

def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件顶层必须是字典")
    if cfg["model"]["name"] != "mlp":
        raise ValueError("本周 model.name 仅支持 mlp")
    if cfg["data"]["name"] != "mnist":
        raise ValueError("本周 data.name 仅支持 mnist")
    if cfg["training"]["dtype"] != "float32":
        raise ValueError("本周 training.dtype 仅支持 float32")

    checkpoint = load_checkpoint(args.checkpoint)
    saved_cfg = checkpoint["config"]
    if cfg["model"] != checkpoint["model_config"]:
        raise ValueError("模型配置与 checkpoint 不一致")
    for field in ("name", "validation_size", "normalize"):
        if cfg["data"][field] != saved_cfg["data"][field]:
            raise ValueError(f"data.{field} 与 checkpoint 不一致")
    if cfg["training"]["seed"] != saved_cfg["training"]["seed"]:
        raise ValueError("training.seed 与 checkpoint 不一致")
    split_sha256 = hashlib.sha256(
        Path(cfg["data"]["split_path"]).read_bytes()
    ).hexdigest()
    if split_sha256 != checkpoint["split_sha256"]:
        raise ValueError("数据划分文件内容与 checkpoint 不一致")

    # 先加载到 CPU，再按本次评估配置选择实际设备。
    device_name = cfg["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_cfg = checkpoint["model_config"]
    model = MLP(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=model_cfg["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    _, val_loader, test_loader = build_mnist_loaders(
        data_dir=cfg["data"]["root"],
        split_path=cfg["data"]["split_path"],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["training"]["seed"],
        validation_size=cfg["data"]["validation_size"],
        num_workers=cfg["data"]["num_workers"],
        mean=cfg["data"]["normalize"]["mean"],
        std=cfg["data"]["normalize"]["std"],
    )
    loader = val_loader if args.split == "val" else test_loader
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)
    assert metrics["n_samples"] == len(loader.dataset)
    print({"split": args.split, "epoch": checkpoint["epoch"], **metrics})
    if args.split == "test":
        mistakes = collect_mistakes(model, loader, device, limit=8)
        result = {
            "split": args.split,
            **metrics,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "config": str(args.config),
            "split_sha256": checkpoint["split_sha256"],
            "mistake_examples_count": len(mistakes),  # 展示数量，不是总错分数。
        }
        experiment_dir = Path(cfg["experiment_dir"])
        result_path = experiment_dir / "test_metrics.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        figure_path = experiment_dir / "figures" / "mistakes.png"
        if not mistakes:
            # 防止把上一次运行的图片误认为本次结果。
            figure_path.unlink(missing_ok=True)
            print("没有错分样例，不生成误分类图。")
            return

        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, mistake in zip(axes.flat, mistakes):
            image = mistake["image"]
            normalize = cfg["data"]["normalize"]
            mean = image.new_tensor(normalize["mean"]).view(-1, 1, 1)
            std = image.new_tensor(normalize["std"]).view(-1, 1, 1)
            image = image * std + mean
            ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"true={mistake['true_label']}, pred={mistake['predicted_label']}\n"
                f"score={mistake['score']:.2%}"
            )
        fig.tight_layout()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path)
        plt.close(fig)
        print("figure saved:", figure_path)

if __name__ == "__main__":
    main()
