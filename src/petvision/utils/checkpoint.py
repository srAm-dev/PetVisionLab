"""读写调用方组装的 checkpoint 字典。"""

from pathlib import Path

import torch


def save_checkpoint(checkpoint, path):
    """将训练调用方组装的 checkpoint 字典写入文件。

    Args:
        checkpoint: 待保存的状态与配置字典。
        path: 输出文件路径。父目录不存在时会创建。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """以 weights_only 模式读取 checkpoint，张量先映射到 CPU。

    Args:
        path: checkpoint 文件路径。

    Returns:
        保存的状态与配置字典。模型由调用者按配置重建。
    """
    return torch.load(path, map_location="cpu", weights_only=True)
