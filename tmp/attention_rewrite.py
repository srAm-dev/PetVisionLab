"""第四周 Day 6：用零 query 和两个头的小例子核对注意力公式。"""

import math

import torch
from torch.nn import functional as F


def attend(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -torch.inf)
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def project_heads(x, weight, heads):
    batch, length, channels = x.shape
    projected = F.linear(x, weight)
    split = projected.reshape(batch, length, heads, channels // heads)
    return split.transpose(1, 2)


def multihead(x, wq, wk, wv, wo, heads, mask=None):
    q = project_heads(x, wq, heads)
    k = project_heads(x, wk, heads)
    v = project_heads(x, wv, heads)
    per_head, weights = attend(q, k, v, mask)
    merged = per_head.transpose(1, 2).reshape_as(x)
    return F.linear(merged, wo), weights


def main():
    dtype = torch.float64
    q = torch.zeros(1, 2, dtype=dtype)
    k = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=dtype)
    v = torch.tensor([[1, 0], [3, 2], [5, 4]], dtype=dtype)
    output, weights = attend(q, k, v)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))
    torch.testing.assert_close(output, torch.tensor([[3, 2]], dtype=dtype))
    print("零 query 的权重、输出", weights.tolist(), output.tolist())

    # 每个头取相邻的四维。位置 1 在两个头中与不同的 key 相似。
    tokens = torch.tensor(
        [
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
        dtype=dtype,
    )
    x = torch.stack((tokens, tokens.flip(0)))
    identity = torch.eye(8, dtype=dtype)
    assert project_heads(x, identity, heads=2).shape == (2, 2, 3, 4)
    causal = torch.ones(3, 3, dtype=torch.bool).triu(1)
    for name, mask in (("双向", None), ("因果", causal)):
        output, weights = multihead(
            x, identity, identity, identity, identity, heads=2, mask=mask
        )
        assert weights.shape == (2, 2, 3, 3) and output.shape == (2, 3, 8)
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
        print(name, "第 0 头位置 1 的权重", weights[0, 0, 1].tolist())
        if mask is None:
            assert (weights[:, :, causal] > 0).all()
            continue
        assert torch.count_nonzero(weights[:, :, causal]) == 0
        torch.testing.assert_close(output[:, 0], x[:, 0])
        # query 1 对前两个 key 的分数为 0 和 1/√4，只能混合这两个 V。
        p = 1 / (1 + math.exp(0.5))
        expected = torch.tensor([p, 1 - p, 0, 0, 1 - p, p, 0, 0], dtype=dtype)
        torch.testing.assert_close(output[0, 1], expected)


if __name__ == "__main__":
    main()
