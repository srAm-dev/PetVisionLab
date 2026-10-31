import argparse
from pathlib import Path
import yaml
import torch
import hashlib
import json
import time
from itertools import islice 
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from petvision.utils.history import save_history, append_history, plot_history
from petvision.data.datasets import build_mnist_loaders, build_cifar10_loaders
from petvision.models.cv.mlp import MLP
from petvision.models.cv.cnn import CNN
from petvision.models.cv.resnet import ResNet18
from petvision.training.cv_trainer import train_one_epoch, evaluate
from petvision.utils.seed import set_seed
from petvision.utils.checkpoint import save_checkpoint

def run_smoke(model, train_loader, val_loader, optimizer, criterion, device):
    train_metrics = train_one_epoch(
        model, islice(train_loader, 2), optimizer, criterion, device
    )
    val_metrics = evaluate(model, islice(val_loader, 2), criterion, device)
    print("smoke train:", train_metrics)
    print("smoke val:", val_metrics)

def run_overfit(model, train_loader, optimizer, criterion, device, n_samples):
    fixed_set = Subset(train_loader.dataset, list(range(n_samples)))
    fixed_loader = DataLoader(fixed_set, batch_size=n_samples, shuffle=False)
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

def run_training(cfg, model, train_loader, val_loader, optimizer, criterion, device):
    best_accuracy = float("-inf")
    split_sha256 = hashlib.sha256(Path(cfg["data"]["split_path"]).read_bytes()).hexdigest()
    epochs = cfg["training"]["epochs"]
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("training.epochs 必须是正整数")

    experiment_dir = Path(cfg["experiment_dir"])
    record_resources = cfg["data"]["name"]=="cifar10"
    variant = experiment_dir.name
    history_path = experiment_dir / "history.csv"
    if record_resources and history_path.exists() and history_path.stat().st_size > 0:
        raise FileExistsError("实验目录已有 history.csv，请先归档原结果或使用新的 experiment_dir")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, allow_unicode=True, sort_keys=False)

    output_dir = Path(cfg["output_dir"])
    if record_resources:
        # 在第一次训练前向之前保存参数和 BN buffers，供 Day 5 对齐初始状态。
        save_checkpoint({
            "model_state": model.state_dict(),
            "model_config": cfg["model"],
            "seed": cfg["training"]["seed"],
            "step": 0,
        }, output_dir / "initial.pt")
    history = []
    if record_resources:
        # 计时包含训练、验证、逐轮日志与 best/last 保存。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()

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
        if record_resources:
            append_history(row, experiment_dir)
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
        save_checkpoint(checkpoint, output_dir / "last.pt")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_checkpoint(checkpoint, output_dir / "best.pt")

    if record_resources:
        # 等待 GPU 完成，结束计时；统计与绘图放在计时范围之外。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else None
        )
        summary = {
            "params": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "train_seconds": train_seconds,
            "peak_allocated_mib": peak_allocated_mib,
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype),
            "timing_scope": "train+validation+epoch_logging+checkpoint_io",
        }
        with (experiment_dir / "run_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        plot_history(history, experiment_dir, variant=variant)
    else:
        save_history(history, experiment_dir)
        plot_history(history, experiment_dir)

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

    data_cfg = cfg["data"]
    if data_cfg["name"]=="mnist":
        loader_builder = build_mnist_loaders
        normalize = cfg["data"]["normalize"]
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ])
        # 当前 MNIST 不做随机增强，训练与评估使用相同的确定性预处理。
        train_transform = eval_transform
    elif data_cfg["name"]=="cifar10":
        assert data_cfg["image_size"]==32
        loader_builder = build_cifar10_loaders
        normalize = cfg["data"]["normalize"]
        train_transform=v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=data_cfg["augmentation"]["random_crop_padding"]),
            v2.RandomHorizontalFlip() if data_cfg["augmentation"]["random_horizontal_flip"] else torch.nn.Identity(),
            v2.Resize((96, 96), antialias=True) if cfg["model"]["name"] == "resnet18" else torch.nn.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalize["mean"], std=normalize["std"])
        ])
        eval_transform=v2.Compose([
            v2.ToImage(),
            v2.Resize((96, 96), antialias=True) if cfg["model"]["name"] == "resnet18" else torch.nn.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalize["mean"], std=normalize["std"])
        ])
    else:
        raise ValueError(f"不支持的数据集:{data_cfg["name"]}")
    if args.overfit_samples is not None:
        train_transform = eval_transform
    train_loader, val_loader, _ = loader_builder(
        data_dir=data_cfg["root"],
        split_path=data_cfg["split_path"],
        batch_size=data_cfg["batch_size"],
        seed=cfg["training"]["seed"],
        train_transform=train_transform,
        eval_transform=eval_transform,
        num_workers=data_cfg["num_workers"],
        validation_size=data_cfg["validation_size"],
    )
    
    model_cfg = cfg["model"]
    if model_cfg["name"]=='mlp':
        model = MLP(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            num_classes=model_cfg["num_classes"],
        ).to(device)
    elif model_cfg["name"]=='cnn':
        model = CNN(
            num_classes=model_cfg["num_classes"],
            use_bn=model_cfg["use_bn"],
        ).to(device)
    elif model_cfg["name"]=='resnet18':
        model = ResNet18(
            num_classes=model_cfg["num_classes"],
        ).to(device)
    else:
        raise ValueError(f"不支持的模型类型:{model_cfg["name"]}")

    training_cfg = cfg["training"]
    criterion = torch.nn.CrossEntropyLoss()
    if training_cfg["optimizer"]=="adam":
        optim = torch.optim.Adam
    elif training_cfg["optimizer"]=="adamw":
        optim = torch.optim.AdamW
    else:
        raise ValueError(f"不支持的优化器类型:{training_cfg["optimizer"]}")
    optimizer = optim(
        model.parameters(), lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    
    if args.smoke:
        run_smoke(model, train_loader, val_loader, optimizer, criterion, device)

    elif args.overfit_samples:
        if args.overfit_samples > len(train_loader.dataset):
            raise ValueError(
                f"--overfit-samples 不能超过训练集数量 {len(train_loader.dataset)}"
            )
        run_overfit(model, train_loader, optimizer, criterion, device, args.overfit_samples)   

    else:
        run_training(cfg, model, train_loader, val_loader, optimizer, criterion, device)

if __name__ == "__main__":
    main()
