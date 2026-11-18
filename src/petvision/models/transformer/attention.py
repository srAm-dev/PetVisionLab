"""实现缩放点积注意力和接受布尔 Mask 的多头自注意力。"""

import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v):
    """计算不带掩码的缩放点积注意力。

    Args:
        q: 查询张量，形状 [..., Tq, d]。
        k: 键张量，形状 [..., Tk, d]。
        v: 值张量，形状 [..., Tk, dv]。

    Returns:
        (output, weights)，形状分别为 [..., Tq, dv] 和 [..., Tq, Tk]。
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，输入和输出均为 [B,T,C]。"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=False):
        """创建 Q/K/V 投影、合头后的输出投影和注意力 dropout。

        Args:
            embed_dim: 每个 token 的总特征数。
        num_heads: 注意力头数，必须整除 embed_dim。
            dropout: softmax 权重的 dropout 概率。
            bias: 四个投影层是否使用偏置。

        Raises:
            ValueError: 特征数或头数不是正整数，或特征数不能被头数整除。
        """
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

    def forward(
        self, x, attn_mask=None, key_padding_mask=None, return_weights=False
    ):
        """对同一输入做多头自注意力。布尔 Mask 中的 True 表示屏蔽。

        Args:
            x: 输入特征，形状 [B, T, C]。
            attn_mask: 可选的 [T, T] 屏蔽表，行对应 query，列对应 key。
            key_padding_mask: 可选的 [B, T] 屏蔽表，标记每条序列的无效 key。
            return_weights: 是否同时返回 dropout 前的每头权重。

        Returns:
            形状 [B, T, C] 的输出。return_weights 为 True 时返回
            (output, weights)，其中 weights 的形状为 [B, heads, T, T]。

        Raises:
            TypeError: Mask 不是布尔 Tensor。
            ValueError: Mask 形状或设备不匹配，或某个 query 的 key 全被屏蔽。
        """
        assert x.ndim == 3, "x应为3维"
        B, T, C = x.shape
        assert C == self.embed_dim

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 先将 C 拆成 heads 和 head_dim，再交换轴得到 [B, heads, T, head_dim]。
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

            blocked_mask = torch.zeros(
                (B, 1, T, T),
                dtype=torch.bool,
                device=scores.device,
            )
            # None 插入长度为 1 的轴，两张表分别按 [1,1,T,T] 和 [B,1,1,T] 广播。
            # |= 用逻辑或合并屏蔽位置，expand_as 再让所有注意力头共用这份规则。
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
