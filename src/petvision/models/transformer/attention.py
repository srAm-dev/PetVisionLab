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
        """Mask 中 True 表示屏蔽，形状分别为 [T,T] 和 [B,T]。"""
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

        if attn_mask is not None or key_padding_mask is not None:
            # 校验两张布尔表，避免错误形状被广播后悄悄参与计算。
            for name, mask, expected_shape in (
                ("attn_mask", attn_mask, (T, T)),
                ("key_padding_mask", key_padding_mask, (B, T)),
            ):
                if mask is None:
                    continue
                if not isinstance(mask, torch.Tensor):
                    raise TypeError(f"{name} 必须是 torch.Tensor")
                if mask.dtype != torch.bool:
                    raise TypeError(f"{name} 必须使用 torch.bool 类型")
                if mask.shape != expected_shape:
                    raise ValueError(f"{name} 形状必须为 {expected_shape}")
                if mask.device != scores.device:
                    raise ValueError(f"{name} 必须与 scores 位于同一设备")

            # [B,1,T,T]：各序列分别合并规则，所有头共用同一张表。
            blocked_mask = torch.zeros(
                (B, 1, T, T), dtype=torch.bool, device=scores.device,
            )
            if attn_mask is not None:
                blocked_mask |= attn_mask[None, None, :, :]
            if key_padding_mask is not None:
                blocked_mask |= key_padding_mask[:, None, None, :]
            blocked_mask = blocked_mask.expand_as(scores)

            # 整行都是 -inf 时 softmax 会产生 NaN，因此先明确拒绝。
            if blocked_mask.all(dim=-1).any().item():
                raise ValueError("合并 Mask 后存在没有可关注 key 的 query")
            scores = scores.masked_fill(blocked_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        final_hidden = self.dropout(weights) @ V
        output = self.out_proj(final_hidden.transpose(1, 2).reshape(B, T, C))

        return (output, weights) if return_weights else output
