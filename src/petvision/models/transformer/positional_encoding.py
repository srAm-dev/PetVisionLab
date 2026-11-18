"""为批次中的序列提供固定的正弦和余弦位置编码。"""

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """为 [B,T,C] 的 token 特征加上固定正弦位置编码。"""

    # pe 由 register_buffer 动态注册，这里声明类型供编辑器识别。
    pe: torch.Tensor

    def __init__(self, embed_dim, max_length=128, dropout=0.0):
        """预计算正弦和余弦位置表，并注册为 pe buffer。

        Args:
            embed_dim: 每个位置的特征数，必须为正偶数。
            max_length: 预计算的位置数，必须为正整数。
            dropout: 特征与位置编码相加后的 dropout 概率。

        Raises:
            ValueError: embed_dim 或 max_length 不符合上述要求。
        """
        super().__init__()
        if type(embed_dim) is not int or embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError("embed_dim 必须为正偶数")
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length 必须为正整数")

        self.embed_dim = embed_dim
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)

        # [max_length,1] / [C/2] 广播得到每个位置、每对通道的角度。
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        channel_indices = torch.arange(0, embed_dim, 2, dtype=torch.float32)
        denominators = 10000 ** (channel_indices / embed_dim)
        angles = positions / denominators

        table = torch.zeros((1, max_length, embed_dim), dtype=torch.float32)
        # 0::2 从通道 0 开始隔一个取一个，1::2 取同一对中的另一列。
        table[0, :, 0::2] = torch.sin(angles)
        table[0, :, 1::2] = torch.cos(angles)
        # buffer 随模型迁移和保存，不作为可学习参数交给优化器。
        self.register_buffer("pe", table)

    def forward(self, x):
        """取当前位置范围的编码并加到输入上。

        Args:
            x: 浮点特征，形状 [B, T, C]，T 不超过 max_length。

        Returns:
            位置相加并经过 dropout 的特征，形状 [B, T, C]。

        Raises:
            ValueError: 输入轴数、特征数或序列长度不符合要求。
        """
        if x.ndim != 3:
            raise ValueError("输入形状必须为 [B,T,C]")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"输入最后一维必须为 embed_dim={self.embed_dim}")
        if x.shape[1] > self.max_length:
            raise ValueError(f"序列长度不能超过 max_length={self.max_length}")

        # 所有 batch 共用位置表；相加保持 [B,T,C]，之后再做 dropout。
        return self.dropout(x + self.pe[:, : x.shape[1]])
