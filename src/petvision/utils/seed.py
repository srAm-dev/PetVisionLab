"""统一设置本项目使用的全局随机种子。"""

import random

import numpy as np
import torch


def set_seed(seed):
    """设置 Python、NumPy 和 PyTorch 的全局随机种子。

    Args:
        seed: 三套随机数生成器共用的整数种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
