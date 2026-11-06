import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，输入和输出均为 [B,T,C]。"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False):
        """创建投影层；embed_dim 是总特征数，num_heads 是头数。"""
        super().__init__()

        if type(embed_dim) is not int or embed_dim <= 0:
            raise ValueError("embed_dim 必须是正整数")
        if type(num_heads) is not int or num_heads <= 0:
            raise ValueError("num_heads 必须是正整数")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 四套独立参数：Q/K/V 投影，以及合头后的输出投影。
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, return_weights=False):
        # 先拒绝尚未支持的 Mask，避免执行投影或消耗 dropout 的随机状态。
        if attn_mask is not None or key_padding_mask is not None:
            raise NotImplementedError("Mask 在 Day 3 接入")

        assert x.ndim == 3, "x应为3维"
        B, T, C = x.shape
        assert C == self.embed_dim

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        final_hidden = self.dropout(weights) @ V
        output = self.out_proj(final_hidden.transpose(1, 2).reshape(B, T, C))

        return (output, weights) if return_weights else output
