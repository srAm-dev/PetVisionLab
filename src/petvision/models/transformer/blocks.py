import torch
from torch import nn
from petvision.models.transformer.normalization import LayerNorm
from petvision.models.transformer.attention import MultiHeadSelfAttention

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout, bias=False)
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim)
        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.addnorm2 = AddNorm(embed_dim, dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attention_output = self.attention(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.addnorm1(x, attention_output)
        ffn_output = self.ffn(x)
        return self.addnorm2(x, ffn_output)
