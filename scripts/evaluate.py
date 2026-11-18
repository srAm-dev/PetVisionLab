"""在独立进程中加载图像分类权重并评估指定划分。"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import v2
import yaml

from petvision.data.datasets import build_cifar10_loaders, build_mnist_loaders
from petvision.models.cv.cnn import CNN
from petvision.models.cv.mlp import MLP
from petvision.models.cv.resnet import ResNet18
from petvision.training.cv_trainer import evaluate
from petvision.utils.checkpoint import load_checkpoint


def collect_mistakes(model, loader, device, limit=8):
    """从已选划分收集前几条分类错误，用于绘图。

    Args:
        model: 返回分类 logits 的模型。
        loader: 提供待检查图片和标签的加载器。
        device: 推理设备。
        limit: 返回的错分样本上限。

    Returns:
        字典列表，每项含 CPU 图片、真实标签、预测标签和预测类别的
        softmax 分数。没有错分样本时为空。
    """
    mistakes = []
    if limit <= 0:
        return mistakes

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)  # [B, 10]
            probabilities = logits.softmax(dim=1)
            predicted = logits.argmax(dim=1)  # [B]

            # nonzero(as_tuple=True) 按轴返回索引。一维布尔表只需取第 0 项。
            wrong_indices = (
                (predicted != labels).nonzero(as_tuple=True)[0].tolist()
            )

            for index in wrong_indices:
                predicted_label = predicted[index].item()

                mistakes.append(
                    {
                        "image": images[index].cpu(),
                        "true_label": labels[index].item(),
                        "predicted_label": predicted_label,
                        "score": probabilities[index, predicted_label].item(),
                    }
                )

                if len(mistakes) >= limit:
                    return mistakes

    return mistakes


def parse_args(argv=None):
    """解析当前脚本的命令行参数。

    Args:
        argv: 可选的参数列表。为 None 时读取实际命令行。

    Returns:
        argparse.Namespace，包含解析后的路径和选项。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    return args


def main():
    """按 checkpoint 重建分类模型，评估选定划分并保存错分样例。"""
    args = parse_args()
    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置文件顶层必须是字典")
    if cfg["training"]["dtype"] != "float32":
        raise ValueError("本周 training.dtype 仅支持 float32")

    checkpoint = load_checkpoint(args.checkpoint)
    saved_cfg = checkpoint["config"]
    data_cfg = cfg["data"]
    requested_model_cfg = cfg["model"].copy()
    saved_model_cfg = checkpoint["model_config"].copy()
    # Day 4 的 ResNet 配置未写开关，语义等同于显式 use_residual=True。
    for model_config in (requested_model_cfg, saved_model_cfg):
        if model_config["name"] == "resnet18":
            model_config.setdefault("use_residual", True)
            if type(model_config["use_residual"]) is not bool:
                raise ValueError("model.use_residual 必须是布尔值")
    if requested_model_cfg != saved_model_cfg:
        raise ValueError("模型配置与 checkpoint 不一致")
    for field in ("name", "validation_size", "normalize"):
        if data_cfg[field] != saved_cfg["data"][field]:
            raise ValueError(f"data.{field} 与 checkpoint 不一致")
    if data_cfg["name"] == "cifar10":
        if data_cfg["image_size"] != saved_cfg["data"]["image_size"]:
            raise ValueError("data.image_size 与 checkpoint 不一致")
    if cfg["training"]["seed"] != saved_cfg["training"]["seed"]:
        raise ValueError("training.seed 与 checkpoint 不一致")
    split_sha256 = hashlib.sha256(
        Path(cfg["data"]["split_path"]).read_bytes()
    ).hexdigest()
    if split_sha256 != checkpoint["split_sha256"]:
        raise ValueError("数据划分文件内容与 checkpoint 不一致")

    device_name = cfg["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_cfg = checkpoint["model_config"]
    if model_cfg["name"] == "mlp":
        model = MLP(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
        )
    elif model_cfg["name"] == "cnn":
        model = CNN(
            num_classes=model_cfg["num_classes"],
            use_bn=model_cfg["use_bn"],
        )
    elif model_cfg["name"] == "resnet18":
        model = ResNet18(
            num_classes=model_cfg["num_classes"],
            use_residual=model_cfg.get("use_residual", True),
        )
    else:
        raise ValueError(f"不支持的模型类型:{model_cfg['name']}")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    if data_cfg["name"] == "mnist":
        loader_builder = build_mnist_loaders
        normalize = data_cfg["normalize"]
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalize["mean"], std=normalize["std"]
                ),
            ]
        )
    elif data_cfg["name"] == "cifar10":
        if data_cfg["image_size"] != 32:
            raise ValueError("CIFAR-10 的 data.image_size 仅支持 32")
        loader_builder = build_cifar10_loaders
        normalize = data_cfg["normalize"]
        eval_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((96, 96), antialias=True)
                if cfg["model"]["name"] == "resnet18"
                else torch.nn.Identity(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
    else:
        raise ValueError(f"不支持的数据集:{data_cfg['name']}")
    _, val_loader, test_loader = loader_builder(
        data_dir=data_cfg["root"],
        split_path=data_cfg["split_path"],
        batch_size=data_cfg["batch_size"],
        seed=cfg["training"]["seed"],
        train_transform=eval_transform,
        eval_transform=eval_transform,
        validation_size=data_cfg["validation_size"],
        num_workers=data_cfg["num_workers"],
    )
    loader = val_loader if args.split == "val" else test_loader
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)
    assert metrics["n_samples"] == len(loader.dataset)
    # **metrics 将指标字典中的键和值展开到当前字典。
    print({"split": args.split, "epoch": checkpoint["epoch"], **metrics})
    # 完整指标统计结束后，再单独收集少量错图用于展示。
    mistakes = collect_mistakes(model, loader, device, limit=8)
    experiment_dir = Path(cfg["experiment_dir"])
    if args.split == "test":
        result = {
            "split": args.split,
            **metrics,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint["epoch"],
            "config": str(args.config),
            "split_sha256": checkpoint["split_sha256"],
            "mistake_examples_count": len(mistakes),  # 展示数量，不是总错分数。
        }
        result_path = experiment_dir / "test_metrics.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    figure_path = experiment_dir / "figures" / f"mistakes_{args.split}.png"
    if not mistakes:
        # 只清除本次 split 的旧错图。
        figure_path.unlink(missing_ok=True)
        print("没有错分样例，不生成误分类图。")
        return

    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    classes = (
        getattr(dataset, "classes", None)
        if data_cfg["name"] == "cifar10"
        else None
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, mistake in zip(axes.flat, mistakes):
        image = mistake["image"]
        mean = image.new_tensor(normalize["mean"]).view(-1, 1, 1)
        std = image.new_tensor(normalize["std"]).view(-1, 1, 1)
        image = image * std + mean
        if data_cfg["name"] == "mnist":
            ax.imshow(
                image.squeeze(0).numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        else:
            ax.imshow(
                image.permute(1, 2, 0).clamp(0, 1).numpy(),
                interpolation="nearest",
            )
        true_label = mistake["true_label"]
        predicted_label = mistake["predicted_label"]
        true_name = (
            classes[true_label] if classes is not None else str(true_label)
        )
        predicted_name = (
            classes[predicted_label]
            if classes is not None
            else str(predicted_label)
        )
        ax.set_title(
            f"true={true_name}\npred={predicted_name}, "
            f"score={mistake['score']:.2%}",
            fontsize=9,
        )
    fig.suptitle(
        f"{experiment_dir.name} | {args.split} | "
        f"epoch={checkpoint['epoch']} | first {len(mistakes)} mistakes"
    )
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figure saved:", figure_path)


if __name__ == "__main__":
    main()
