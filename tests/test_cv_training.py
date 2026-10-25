import torch

from petvision.models.cv.mlp import MLP


def test_mlp_forward_backward():
    for batch_size in (1, 4):
        # 每次新建模型，分别检查单张图片和一个小批次。
        model = MLP().cpu()
        images = torch.randn(batch_size, 1, 28, 28)
        labels = torch.randint(0, 10, (batch_size,), dtype=torch.long)

        # 前向：每张图片得到 10 个类别的 logits。
        model.zero_grad()
        logits = model(images)
        assert logits.shape == (batch_size, 10)

        # 损失：将 logits 与正确类别编号比较，得到一个有限标量。
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        # 反向：为参与计算的模型参数计算梯度。
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"

        print(f"batch={batch_size}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")

def test_mlp_parameter_count():
    model = MLP()
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters:", parameter_count)
    assert parameter_count == 235146
    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape))

def test_mlp_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    images = torch.randn(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,), dtype=torch.long).to(device)

    logits = model(images)
    assert logits.shape == (4, 10)
    assert logits.device == images.device == labels.device
    assert all(parameter.device == images.device for parameter in model.parameters())
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss).item()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"
    print("device:", logits.device)


def test_evaluate_metrics():
    from itertools import islice

    from torch.utils.data import DataLoader, TensorDataset
    from petvision.training.cv_trainer import evaluate

    # 为了单独检查统计，直接准备 3 条已知的分类分数。
    logits = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
    model = torch.nn.Identity()
    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))
    expected_loss = criterion(logits, labels).item()
    assert metrics["n_samples"] == 3
    assert abs(metrics["loss"] - expected_loss) < 1e-6
    assert abs(metrics["accuracy"] - 1 / 3) < 1e-6

    # 只读取第一批时，应统计 2 条，而不是整个 dataset 的 3 条。
    partial = evaluate(model, islice(loader, 1), criterion, torch.device("cpu"))
    assert partial["n_samples"] == 2
    assert abs(partial["loss"] - criterion(logits[:2], labels[:2]).item()) < 1e-6
    assert partial["accuracy"] == 0.5
    print("full:", metrics)
    print("first batch:", partial)