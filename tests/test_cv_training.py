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


def test_checkpoint_roundtrip(tmp_path):
    from petvision.utils.checkpoint import load_checkpoint, save_checkpoint

    model_config = {"input_dim": 4, "hidden_dims": [8], "num_classes": 3}
    model = MLP(**model_config).cpu().eval()
    images = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    with torch.no_grad():
        logits_before = model(images)
        loss_before = torch.nn.functional.cross_entropy(logits_before, labels)

    path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint({
        "model_state": model.state_dict(),
        "model_config": model_config,
        "epoch": 2,
    }, path)
    checkpoint = load_checkpoint(path)
    restored = MLP(**checkpoint["model_config"]).cpu().eval()
    restored.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        logits_after = restored(images)
        loss_after = torch.nn.functional.cross_entropy(logits_after, labels)

    assert checkpoint["epoch"] == 2
    assert all(value.device.type == "cpu" for value in checkpoint["model_state"].values())
    torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=1e-6)
    torch.testing.assert_close(loss_after, loss_before, rtol=0, atol=1e-6)
    assert torch.equal(logits_after.argmax(1), logits_before.argmax(1))
    print("checkpoint logits max error:", (logits_after - logits_before).abs().max().item())


def test_checkpoint_best_and_last(tmp_path, monkeypatch):
    import argparse
    import hashlib
    from pathlib import Path

    import yaml
    from petvision.utils.checkpoint import load_checkpoint

    # 调用实际训练入口，只替换数据来源与验证分数，检查真正写出的文件。
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    import train_cv

    cfg = yaml.safe_load((project_root / "configs/cv/mlp_mnist.yaml").read_text())
    cfg["model"].update(input_dim=4, hidden_dims=[8], num_classes=3)
    cfg["training"].update(device="cpu", epochs=4)
    cfg["experiment_dir"] = str(tmp_path / "experiment")
    cfg["output_dir"] = str(tmp_path / "weights")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    cfg["data"]["split_path"] = str(split_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(train_cv, "parse_args", lambda: argparse.Namespace(
        config=config_path, smoke=False, overfit_samples=None,
    ))
    batch = [(torch.randn(3, 4), torch.tensor([0, 1, 2], dtype=torch.long))]
    monkeypatch.setattr(train_cv, "build_mnist_loaders", lambda **kwargs: (batch, batch, batch))
    accuracies = iter([0.80, 0.85, 0.82, 0.85])
    monkeypatch.setattr(train_cv, "evaluate", lambda *args: {
        "loss": 0.5, "accuracy": next(accuracies), "n_samples": 3,
    })
    monkeypatch.setattr(train_cv, "plot_history", lambda *args: None)
    train_cv.main()

    best = load_checkpoint(tmp_path / "weights" / "best.pt")
    last = load_checkpoint(tmp_path / "weights" / "last.pt")
    assert best["epoch"] == 2  # 第 4 轮持平，保留较早一次。
    assert last["epoch"] == 4
    assert best["val_accuracy"] == 0.85
    assert best["config"] == cfg
    assert best["model_config"] == cfg["model"]
    assert best["split_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
    assert any(not torch.equal(best["model_state"][key], value)
               for key, value in last["model_state"].items())
