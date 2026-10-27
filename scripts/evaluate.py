import yaml
import hashlib
import torch
import argparse
from pathlib import Path
from petvision.models.cv.mlp import MLP
from petvision.training.cv_trainer import evaluate
from petvision.utils.checkpoint import load_checkpoint
from petvision.data.datasets import build_mnist_loaders


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

if __name__ == "__main__":
    main()
