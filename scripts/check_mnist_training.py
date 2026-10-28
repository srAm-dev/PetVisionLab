from itertools import islice
import math

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from petvision.data.datasets import build_mnist_loaders
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import train_one_epoch, evaluate
from petvision.utils.seed import set_seed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_loader, val_loader, _ = build_mnist_loaders(
        data_dir="datasets/raw",
        split_path="experiments/mnist_mlp/week2/mnist_split.json",
        batch_size=128,
        seed=42,
        train_transform=eval_transform,
        eval_transform=eval_transform,
    )
    criterion = torch.nn.CrossEntropyLoss()
    print("device:", device)

    # A. smoke：只训练两批、验证两批，确认接口能接通。
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_metrics = train_one_epoch(
        model, islice(train_loader, 2), optimizer, criterion, device
    )
    val_metrics = evaluate(model, islice(val_loader, 2), criterion, device)
    print("smoke train:", train_metrics)
    print("smoke val:", val_metrics)
    assert train_metrics["n_samples"] == 256
    assert val_metrics["n_samples"] == 256
    assert math.isfinite(train_metrics["loss"])
    assert math.isfinite(val_metrics["loss"])

    # B. 过拟合：新建模型和优化器，反复学习同样的 64 条训练样本。
    set_seed(42)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    fixed_set = Subset(train_loader.dataset, list(range(64)))
    fixed_loader = DataLoader(fixed_set, batch_size=64, shuffle=False)
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
    assert final["n_samples"] == 64
    assert final["loss"] < initial["loss"]
    assert final["accuracy"] >= 0.98, "固定小样本尚未拟合好，请检查训练流程"


if __name__ == "__main__":
    main()
