"""实现供 Transformer 子层复用的末轴归一化。"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """对每个位置的最后一维归一化，再应用可学习的缩放和偏移。"""

    def __init__(self, embed_dim, eps=1e-5):
        """创建缩放参数 gamma 和偏移参数 beta。

        Args:
            embed_dim: 末轴特征数。
            eps: 加在方差上的小量，避免除以零。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        """在最后一维计算均值和按特征数平均的方差。

        Args:
            x: 末轴大小为 embed_dim 的张量。

        Returns:
            与 x 同形状的归一化张量。

        Raises:
            ValueError: x 为零维张量或末轴大小不匹配。
        """
        if x.ndim == 0 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                "LayerNorm 输入至少为一维，且最后一维必须为 "
                f"embed_dim={self.embed_dim}，"
                f"实际形状为 {tuple(x.shape)}"
            )
        # keepdim 保留长度为 1 的末轴，均值和方差可以广播回每个特征。
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        return y
