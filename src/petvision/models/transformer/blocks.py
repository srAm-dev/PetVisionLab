"""组,位置编码、残差归一化和前馈网络，构建教学 Encoder。"""

import math

import torch
from torch import nn

from petvision.models.transformer.attention import MultiHeadSelfAttention
from petvision.models.transformer.normalization import LayerNorm
from petvision.models.transformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)


class FeedForward(nn.Module):
    """对每个 token 独立执行 Linear → ReLU → Linear。

    输入和输出均为 [B, T, C]。
    """

    def __init__(self, embed_dim, hidden_dim):
        """构造逐位置前馈网络。

        Args:
            embed_dim: 输入和输出的特征数。
            hidden_dim: 中间层的特征数。
        """
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """对分支输出做 dropout，与残差相加后沿最后一维归一化。"""

    def __init__(self, embed_dim, dropout):
        """创建分支 dropout 和 LayerNorm。

        Args:
            embed_dim: 最后一维的特征数。
            dropout: 分支输出的 dropout 概率。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x, y):
        """合并残差和子层输出。

        Args:
            x: 残差路径的输入特征。
            y: 子层输出，与 x 形状相同。

        Returns:
            与 x 同形状的归一化结果。
        """
        return self.norm(x + self.dropout(y))


class EncoderBlock(nn.Module):
    """按注意力、AddNorm、FFN、AddNorm 的顺序处理序列。"""

    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.0):
        """创建多头注意力、前馈网络及两套独立的 AddNorm。

        Args:
            embed_dim: 每个 token 的特征数。
            num_heads: 注意力头数。
            ffn_hidden_dim: 前馈网络的中间层宽度。
            dropout: 注意力权重和两个 AddNorm 使用的 dropout 概率。
        """
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim, num_heads, dropout=dropout, bias=False
        )
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim)
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """编码一批输入序列。

        Args:
            x: 形状 [B, T, C] 的浮点特征。
            attn_mask: 可选的 [T, T] 布尔表，True 屏蔽一对 query / key。
            key_padding_mask: 可选的 [B, T] 布尔表，True 屏蔽无效 key。

        Returns:
            形状 [B, T, C] 的输出特征。
        """
        attention_output = self.attention(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.addnorm1(x, attention_output)
        ffn_output = self.ffn(x)
        return self.addnorm2(x, ffn_output)


class Encoder(nn.Module):
    """缩放输入并加入位置编码，再依次执行多个 post-LN Block。"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_hidden_dim,
        depth=2,
        dropout=0.0,
        max_length=128,
    ):
        """创建位置编码模块和独立的 EncoderBlock。

        Args:
            embed_dim: 输入特征数，须为正偶数且能被 num_heads 整除。
            num_heads: 各层的注意力头数。
            ffn_hidden_dim: 各层前馈网络的中间层宽度。
            depth: EncoderBlock 的数量。
            dropout: 位置编码和各层使用的 dropout 概率。
            max_length: 位置表支持的最大序列长度。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.position = SinusoidalPositionalEncoding(
            embed_dim=embed_dim,
            max_length=max_length,
            dropout=dropout,
        )
        # 每次循环新建一个 Block，ModuleList 将各层注册为 Encoder 的子模块。
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """编码一批输入序列。

        Args:
            x: 形状 [B, T, C] 的浮点特征。
            attn_mask: 可选的 [T, T] 布尔表，True 屏蔽一对 query / key。
            key_padding_mask: 可选的 [B, T] 布尔表，True 屏蔽无效 key。

        Returns:
            形状 [B, T, C] 的输出特征。
        """
        x = self.position(x * math.sqrt(self.embed_dim))
        for block in self.blocks:
            x = block(x, attn_mask, key_padding_mask)
        return x
