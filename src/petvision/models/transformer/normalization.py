import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"LayerNorm 输入至少为一维，且最后一维必须为 embed_dim={self.embed_dim}，"
                f"实际形状为 {tuple(x.shape)}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        y    = (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        return y
