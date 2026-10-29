import pytest
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
    assert not (tmp_path / "weights" / "initial.pt").exists()
    assert not (tmp_path / "experiment" / "run_summary.json").exists()


def test_cifar_initial_checkpoint_precedes_training(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import yaml
    from petvision.models.cv.cnn import CNN
    from petvision.utils.checkpoint import load_checkpoint

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    import train_cv

    cfg = yaml.safe_load((project_root / "configs/cv/cnn_cifar10.yaml").read_text())
    cfg["training"].update(device="cpu", epochs=1)
    cfg["experiment_dir"] = str(tmp_path / "experiment")
    cfg["output_dir"] = str(tmp_path / "weights")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    cfg["data"]["split_path"] = str(split_path)
    model = CNN(use_bn=True).cpu()
    batch = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    initial_path = tmp_path / "weights" / "initial.pt"
    initial_state = {}
    original_train = train_cv.train_one_epoch

    def check_initial_and_train(model, loader, optimizer, criterion, device):
        # 第一轮训练尚未开始时，文件已经存在，且与当前模型完全一致。
        assert initial_path.is_file()
        initial = load_checkpoint(initial_path)
        assert initial["step"] == 0
        assert initial["model_config"] == cfg["model"]
        assert initial["seed"] == cfg["training"]["seed"]
        initial_state.update(initial["model_state"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                assert torch.equal(layer.running_mean, torch.zeros_like(layer.running_mean))
                assert torch.equal(layer.running_var, torch.ones_like(layer.running_var))
                assert layer.num_batches_tracked.item() == 0
        return original_train(model, loader, optimizer, criterion, device)

    monkeypatch.setattr(train_cv, "train_one_epoch", check_initial_and_train)
    train_cv.run_training(cfg, model, batch, batch, optimizer, criterion, torch.device("cpu"))

    # 完成真实的参数更新后，initial 仍保留训练前的参数与 BN 状态。
    initial = load_checkpoint(initial_path)
    last = load_checkpoint(tmp_path / "weights" / "last.pt")
    for name, value in initial_state.items():
        torch.testing.assert_close(initial["model_state"][name], value, rtol=0, atol=0)
    assert any(
        not torch.equal(initial_state[name], last["model_state"][name])
        for name, _ in model.named_parameters()
    )
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            assert layer.num_batches_tracked.item() == 1
    experiment_dir = Path(cfg["experiment_dir"])
    summary = json.loads((experiment_dir / "run_summary.json").read_text())
    assert summary["params"] == sum(parameter.numel() for parameter in model.parameters())
    assert summary["epochs"] == 1
    assert summary["train_seconds"] >= 0
    assert summary["peak_allocated_mib"] is None
    assert summary["device"] == "cpu" and summary["dtype"] == "torch.float32"
    assert (experiment_dir / "figures" / "training_curves.png").is_file()


def test_cifar_check_modes_do_not_save_initial(tmp_path, monkeypatch):
    import argparse
    from pathlib import Path

    import yaml
    from torch.utils.data import DataLoader, TensorDataset

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    import train_cv

    cfg = yaml.safe_load((project_root / "configs/cv/cnn_cifar10.yaml").read_text())
    cfg["training"].update(device="cpu", epochs=1)
    cfg["experiment_dir"] = str(tmp_path / "experiment")
    cfg["output_dir"] = str(tmp_path / "weights")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    cfg["data"]["split_path"] = str(split_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loader = DataLoader(TensorDataset(
        torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]),
    ), batch_size=2)
    monkeypatch.setattr(train_cv, "build_cifar10_loaders", lambda **kwargs: (loader, loader, loader))
    train_calls = []
    metrics = {"loss": 0.1, "accuracy": 1.0, "n_samples": 2}

    def fake_train(*args):
        train_calls.append(1)
        return metrics.copy()

    # 只检查运行模式与文件写入边界，跳过耗时的参数更新。
    monkeypatch.setattr(train_cv, "train_one_epoch", fake_train)
    monkeypatch.setattr(train_cv, "evaluate", lambda *args: metrics.copy())
    for smoke, overfit_samples, expected_calls in ((True, None, 1), (False, 2, 200)):
        args = argparse.Namespace(config=config_path, smoke=smoke, overfit_samples=overfit_samples)
        monkeypatch.setattr(train_cv, "parse_args", lambda: args)
        train_calls.clear()
        train_cv.main()
        assert len(train_calls) == expected_calls
        assert not Path(cfg["output_dir"]).exists()
        assert not Path(cfg["experiment_dir"]).exists()


def test_history_save_and_append_use_same_columns(tmp_path):
    import csv

    from petvision.utils.history import save_history, append_history

    history = [
        {"epoch": 1, "train_loss": 0.8, "train_accuracy": 0.6,
         "val_loss": 0.9, "val_accuracy": 0.5},
        {"epoch": 2, "train_loss": 0.6, "train_accuracy": 0.7,
         "val_loss": 0.8, "val_accuracy": 0.6},
    ]
    saved_dir = tmp_path / "saved"
    appended_dir = tmp_path / "appended"
    save_history(history, saved_dir)
    for row in history:
        append_history(row, appended_dir)

    with (saved_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        saved_rows = list(csv.reader(file))
    with (appended_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        appended_rows = list(csv.reader(file))
    assert saved_rows == appended_rows
    assert saved_rows[0] == [
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    ]
    assert len(saved_rows) == 3

    # 向一次性保存的文件追加时，列仍对齐，且不会再次写入表头。
    append_history({**history[-1], "epoch": 3}, saved_dir)
    with (saved_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        combined_rows = list(csv.reader(file))
    assert combined_rows[:3] == saved_rows
    assert combined_rows[-1][0] == "3"
    assert len(combined_rows) == 4


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
def test_cifar_resource_logging(tmp_path, monkeypatch, device_name):
    import csv
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import yaml
    from petvision.models.cv.cnn import CNN

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    import train_cv

    cfg = yaml.safe_load((project_root / "configs/cv/cnn_cifar10.yaml").read_text())
    cfg["training"].update(device=device_name, epochs=2)
    experiment_dir = tmp_path / "cnn_bn_on"
    cfg["experiment_dir"] = str(experiment_dir)
    cfg["output_dir"] = str(tmp_path / "weights")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    cfg["data"]["split_path"] = str(split_path)
    model = CNN().cpu()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    events = []
    epoch = 0
    timer_values = iter((10.0, 16.5))
    original_save = train_cv.save_checkpoint
    original_append = train_cv.append_history

    def read_timer():
        events.append("timer")
        return next(timer_values)

    def synchronize(device):
        assert device_name == "cuda" and device.type == "cuda"
        events.append("sync")

    def reset_peak(device):
        assert device_name == "cuda" and device.type == "cuda"
        events.append("reset")

    def peak_memory(device):
        assert device_name == "cuda" and device.type == "cuda"
        events.append("peak")
        return 4 * 1024**2

    def train(*args):
        nonlocal epoch
        epoch += 1
        if epoch == 2:
            # 第二轮开始前，第一轮的记录已经落盘。
            with (experiment_dir / "history.csv").open(newline="", encoding="utf-8") as file:
                assert [row["epoch"] for row in csv.DictReader(file)] == ["1"]
            assert not (experiment_dir / "run_summary.json").exists()
        events.append(f"train{epoch}")
        return {"loss": 0.8, "accuracy": 0.5, "n_samples": 2}

    def evaluate(*args):
        events.append(f"val{epoch}")
        return {"loss": 0.9, "accuracy": 0.5, "n_samples": 2}

    def append(row, directory):
        events.append(f"log{row['epoch']}")
        original_append(row, directory)

    def save(checkpoint, path):
        events.append(Path(path).name)
        original_save(checkpoint, path)

    def plot(history, directory, variant=None):
        events.append("plot")
        assert len(history) == 2 and variant == "cnn_bn_on"
        summary = json.loads((Path(directory) / "run_summary.json").read_text())
        assert summary["params"] == 94986 and summary["epochs"] == 2
        assert summary["train_seconds"] == 6.5
        assert summary["peak_allocated_mib"] == (4.0 if device_name == "cuda" else None)
        assert summary["device"] == device_name and summary["dtype"] == "torch.float32"
        assert summary["timing_scope"] == "train+validation+epoch_logging+checkpoint_io"

    # 使用 CPU 模型和模拟 CUDA 接口，验证时序与单位换算，无需 GPU。
    monkeypatch.setattr(train_cv, "time", SimpleNamespace(perf_counter=read_timer))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak_memory)
    monkeypatch.setattr(train_cv, "train_one_epoch", train)
    monkeypatch.setattr(train_cv, "evaluate", evaluate)
    monkeypatch.setattr(train_cv, "append_history", append)
    monkeypatch.setattr(train_cv, "save_checkpoint", save)
    monkeypatch.setattr(train_cv, "plot_history", plot)
    monkeypatch.setattr(train_cv, "save_history", lambda *args: pytest.fail("不能覆盖逐轮追加的日志"))
    train_cv.run_training(cfg, model, [], [], optimizer, criterion, torch.device(device_name))

    with (experiment_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    assert [row[0] for row in rows[1:]] == ["1", "2"]
    expected_events = ["initial.pt"]
    if device_name == "cuda":
        expected_events += ["sync", "reset"]
    expected_events += [
        "timer", "train1", "val1", "log1", "last.pt", "best.pt",
        "train2", "val2", "log2", "last.pt",
    ]
    if device_name == "cuda":
        expected_events.append("sync")
    expected_events.append("timer")
    if device_name == "cuda":
        expected_events.append("peak")
    expected_events.append("plot")
    assert events == expected_events


def test_cifar_rejects_existing_history_before_overwriting_outputs(tmp_path, monkeypatch):
    from pathlib import Path

    import yaml
    from petvision.models.cv.cnn import CNN

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root / "scripts"))
    import train_cv

    cfg = yaml.safe_load((project_root / "configs/cv/cnn_cifar10.yaml").read_text())
    cfg["training"].update(device="cpu", epochs=1)
    experiment_dir = tmp_path / "experiment"
    cfg["experiment_dir"] = str(experiment_dir)
    cfg["output_dir"] = str(tmp_path / "weights")
    split_path = tmp_path / "split.json"
    split_path.write_text("{}", encoding="utf-8")
    cfg["data"]["split_path"] = str(split_path)
    existing_files = {
        experiment_dir / "history.csv": b"existing history",
        experiment_dir / "config.yaml": b"existing config",
        tmp_path / "weights" / "initial.pt": b"existing initial",
    }
    for path, content in existing_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    model = CNN().cpu()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(FileExistsError, match="history.csv"):
        train_cv.run_training(
            cfg, model, [], [], optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
        )
    for path, content in existing_files.items():
        assert path.read_bytes() == content
