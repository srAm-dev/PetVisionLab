from pathlib import Path
import torch


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)
