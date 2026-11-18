"""为 MNIST 和 CIFAR-10 复用固定划分，由调用方提供预处理。"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST


def build_mnist_loaders(
    data_dir,
    split_path,
    batch_size,
    seed,
    train_transform,
    eval_transform,
    validation_size=5000,
    num_workers=0,
):
    """创建 MNIST 的训练、验证与官方测试加载器。

    Args:
        data_dir: 数据缓存根目录。
        split_path: 保存或复用训练与验证索引的 JSON 路径。
        batch_size: 每批最多包含的样本数。
        seed: 划分索引和训练批次顺序使用的种子。
        train_transform: 训练图片的预处理。
        eval_transform: 验证与测试图片的预处理，通常是确定性的。
        validation_size: 从官方训练集划出的验证样本数。
        num_workers: DataLoader 使用的工作进程数。

    Returns:
        (train_loader, val_loader, test_loader)。训练顺序会打乱，
        三个加载器都保留最后一个不足整批的批次。

    Raises:
        ValueError: 训练集数量、验证集大小或保存的划分与当前设置不符。
    """
    train_base = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    # 两个实例读取同一训练文件，各自保存 transform，避免训练增强进入验证集。
    val_base = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_set = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )
    split_path = Path(split_path)
    n_total = len(train_base)
    if n_total != 60000:
        raise ValueError(f"MNIST 官方训练集应有 60000 条，实际为 {n_total}")

    if type(validation_size) is not int or not 0 < validation_size < n_total:
        raise ValueError("validation_size 必须是大于 0、小于总样本数的整数")

    n_train = n_total - validation_size

    # 首次生成划分；已有文件时读取，避免覆盖之前的实验条件。
    split_exists = split_path.exists()
    if split_exists:
        with split_path.open("r", encoding="utf-8") as file:
            split_data = json.load(file)
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()
        split_data = {
            "dataset": "mnist",
            "version": 1,
            "seed": seed,
            "n_total": n_total,
            "train_indices": indices[:n_train],
            "val_indices": indices[n_train:],
        }

    # 读取和首次生成的划分都使用相同的完整性检查。
    if not isinstance(split_data, dict):
        raise ValueError("划分 JSON 的顶层必须是字典")
    expected_metadata = {
        "dataset": "mnist",
        "version": 1,
        "seed": seed,
        "n_total": n_total,
    }
    for name, expected in expected_metadata.items():
        if split_data.get(name) != expected:
            raise ValueError(
                f"划分文件的 {name} 与当前设置不符："
                f"期望 {expected!r}，实际 {split_data.get(name)!r}"
            )

    for name, expected_size in (
        ("train_indices", n_train),
        ("val_indices", validation_size),
    ):
        values = split_data.get(name)
        if not isinstance(values, list):
            raise ValueError(f"{name} 必须是列表")
        if len(values) != expected_size:
            raise ValueError(
                f"{name} 应有 {expected_size} 项，实际为 {len(values)}"
            )
        if not all(type(index) is int for index in values):
            raise ValueError(f"{name} 中的索引必须是整数")
        if not all(0 <= index < n_total for index in values):
            raise ValueError(f"{name} 中存在超出 0..{n_total - 1} 的索引")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} 中存在重复索引")

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    train_index_set = set(train_indices)
    val_index_set = set(val_indices)
    # & 取交集，| 取并集。训练和验证应互不重叠，合起来覆盖官方训练集。
    if train_index_set & val_index_set:
        raise ValueError("训练集与验证集的索引存在重叠")
    if train_index_set | val_index_set != set(range(n_total)):
        raise ValueError("训练集与验证集的索引未完整覆盖官方训练集")

    # 只有首次生成且检查通过后，才保存到约定路径。
    if not split_exists:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as file:
            json.dump(split_data, file, ensure_ascii=False, indent=2)

    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def build_cifar10_loaders(
    data_dir,
    split_path,
    batch_size,
    seed,
    train_transform,
    eval_transform,
    validation_size=5000,
    num_workers=0,
):
    """创建 CIFAR-10 的训练、验证与官方测试加载器。

    Args:
        data_dir: 数据缓存根目录。
        split_path: 保存或复用训练与验证索引的 JSON 路径。
        batch_size: 每批最多包含的样本数。
        seed: 划分索引和训练批次顺序使用的种子。
        train_transform: 训练图片的预处理。
        eval_transform: 验证与测试图片的预处理，通常是确定性的。
        validation_size: 从官方训练集划出的验证样本数。
        num_workers: DataLoader 使用的工作进程数。

    Returns:
        (train_loader, val_loader, test_loader)。训练顺序会打乱，
        三个加载器都保留最后一个不足整批的批次。

    Raises:
        ValueError: 训练集数量、验证集大小或保存的划分与当前设置不符。
    """
    train_base = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_base = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_set = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )
    split_path = Path(split_path)
    n_total = len(train_base)
    if n_total != 50000:
        raise ValueError(f"CIFAR10 官方训练集应有 50000 条，实际为 {n_total}")

    if type(validation_size) is not int or not 0 < validation_size < n_total:
        raise ValueError("validation_size 必须是大于 0、小于总样本数的整数")

    n_train = n_total - validation_size

    # 首次生成划分；已有文件时读取，避免覆盖之前的实验条件。
    split_exists = split_path.exists()
    if split_exists:
        with split_path.open("r", encoding="utf-8") as file:
            split_data = json.load(file)
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()
        split_data = {
            "dataset": "cifar10",
            "version": 1,
            "seed": seed,
            "n_total": n_total,
            "train_indices": indices[:n_train],
            "val_indices": indices[n_train:],
        }

    # 读取和首次生成的划分都使用相同的完整性检查。
    if not isinstance(split_data, dict):
        raise ValueError("划分 JSON 的顶层必须是字典")
    expected_metadata = {
        "dataset": "cifar10",
        "version": 1,
        "seed": seed,
        "n_total": n_total,
    }
    for name, expected in expected_metadata.items():
        if split_data.get(name) != expected:
            raise ValueError(
                f"划分文件的 {name} 与当前设置不符："
                f"期望 {expected!r}，实际 {split_data.get(name)!r}"
            )

    for name, expected_size in (
        ("train_indices", n_train),
        ("val_indices", validation_size),
    ):
        values = split_data.get(name)
        if not isinstance(values, list):
            raise ValueError(f"{name} 必须是列表")
        if len(values) != expected_size:
            raise ValueError(
                f"{name} 应有 {expected_size} 项，实际为 {len(values)}"
            )
        if not all(type(index) is int for index in values):
            raise ValueError(f"{name} 中的索引必须是整数")
        if not all(0 <= index < n_total for index in values):
            raise ValueError(f"{name} 中存在超出 0..{n_total - 1} 的索引")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} 中存在重复索引")

    train_indices = split_data["train_indices"]
    val_indices = split_data["val_indices"]
    train_index_set = set(train_indices)
    val_index_set = set(val_indices)
    if train_index_set & val_index_set:
        raise ValueError("训练集与验证集的索引存在重叠")
    if train_index_set | val_index_set != set(range(n_total)):
        raise ValueError("训练集与验证集的索引未完整覆盖官方训练集")

    # 只有首次生成且检查通过后，才保存到约定路径。
    if not split_exists:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as file:
            json.dump(split_data, file, ensure_ascii=False, indent=2)

    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, val_indices)

    # 第四步：分别创建训练、验证和测试 DataLoader。
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
