import argparse
from pathlib import Path
import yaml
import torch
import csv
import hashlib
import matplotlib.pyplot as plt
from itertools import islice 
from torch.utils.data import Subset, DataLoader
from petvision.data.datasets import build_mnist_loaders
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import train_one_epoch, evaluate
from petvision.utils.seed import set_seed
from petvision.utils.checkpoint import save_checkpoint


def save_history(history, experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy",
    ]
    with (experiment_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

def plot_history(history, experiment_dir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set(title="Loss", ylabel="Cross entropy")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val")
    axes[1].set(title="Accuracy", ylabel="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()

    figure_dir = Path(experiment_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "training_curves.png", dpi=150)
    plt.close(fig)

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--overfit-samples", type=int)
    args = parser.parse_args(argv)
    if args.overfit_samples is not None and args.overfit_samples <= 0:
        parser.error("--overfit-samples 必须大于 0")
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
    if cfg["training"]["optimizer"] != "adam":
        raise ValueError("本周 training.optimizer 仅支持 adam")
    if cfg["training"]["dtype"] != "float32":
        raise ValueError("本周 training.dtype 仅支持 float32")
    print(cfg["model"]["name"], cfg["data"]["batch_size"])
    print(args.smoke, args.overfit_samples)
    # 按配置创建数据、模型和优化器，再进入所选运行模式。
    set_seed(cfg["training"]["seed"])
    device_name = cfg["training"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_loader, val_loader, _ = build_mnist_loaders(
        data_dir=cfg["data"]["root"],
        split_path=cfg["data"]["split_path"],
        batch_size=cfg["data"]["batch_size"],
        seed=cfg["training"]["seed"],
        validation_size=cfg["data"]["validation_size"],
        num_workers=cfg["data"]["num_workers"],
        mean=cfg["data"]["normalize"]["mean"],
        std=cfg["data"]["normalize"]["std"],
    )
    model_cfg = cfg["model"]
    model = MLP(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    if args.smoke:
        train_metrics = train_one_epoch(
            model, islice(train_loader, 2), optimizer, criterion, device
        )
        val_metrics = evaluate(model, islice(val_loader, 2), criterion, device)
        print("smoke train:", train_metrics)
        print("smoke val:", val_metrics)

    elif args.overfit_samples:
        if args.overfit_samples > len(train_loader.dataset):
            raise ValueError(
                f"--overfit-samples 不能超过训练集数量 {len(train_loader.dataset)}"
            )
        fixed_set = Subset(train_loader.dataset, list(range(args.overfit_samples)))
        fixed_loader = DataLoader(fixed_set, batch_size=args.overfit_samples, shuffle=False)
        images, labels = next(iter(fixed_loader))
        fixed_batch = [(images, labels)]
        initial = evaluate(model, fixed_batch, criterion, device)
        print("overfit initial:", initial)
        for step in range(1, 201):
            train_one_epoch(model, fixed_batch, optimizer, criterion, device)
            if step % 20 == 0:
                metrics = evaluate(model, fixed_batch, criterion, device)
                print(
                    f"step={step}, loss={metrics['loss']:.6f}, "
                    f"accuracy={metrics['accuracy']:.2%}"
                )
    
        final = evaluate(model, fixed_batch, criterion, device)
        print("overfit final:", final)

    else:
        best_accuracy = float("-inf")
        split_sha256 = hashlib.sha256(Path(cfg["data"]["split_path"]).read_bytes()).hexdigest()
        epochs = cfg["training"]["epochs"]
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("training.epochs 必须是正整数")

        experiment_dir = Path(cfg["experiment_dir"])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file, allow_unicode=True, sort_keys=False)

        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)
            print(
                f"epoch={epoch}, "
                f"train_loss={row['train_loss']:.6f}, "
                f"train_accuracy={row['train_accuracy']:.2%}, "
                f"val_loss={row['val_loss']:.6f}, "
                f"val_accuracy={row['val_accuracy']:.2%}"
            )
            # 每轮保存 last；仅验证准确率提升时更新 best。
            checkpoint = {
                'model_state': model.state_dict(),
                'model_config': cfg['model'],
                'config': cfg,
                'epoch': epoch,
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'seed': cfg['training']['seed'],
                'split_sha256': split_sha256,
            }
            output_dir = Path(cfg["output_dir"])
            save_checkpoint(checkpoint, output_dir / "last.pt")
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                save_checkpoint(checkpoint, output_dir / "best.pt")

        save_history(history, experiment_dir)
        plot_history(history, experiment_dir)

        


if __name__ == "__main__":
    main()
