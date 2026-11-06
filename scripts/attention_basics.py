"""第四周 Day 1：用手算、小张量和梯度检查理解缩放点积 Attention。"""
import math
import torch
from petvision.models.transformer.attention import scaled_dot_product_attention


def main():
    # 使用 CPU float64，固定随机流，方便重复运行并核对结果。
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(42)

    print("1. 手算例子：零 query 得到均匀权重")
    q = torch.zeros((1, 2), dtype=dtype)
    k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype)
    v = torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=dtype)

    # 函数返回 output、weights；scores 在这里另算一次，便于查看中间量。
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    output, weights = scaled_dot_product_attention(q, k, v)

    print("Q:", q)
    print("K:", k)
    print("V:", v)
    print("K 转置前后形状:", tuple(k.shape), "->", tuple(k.transpose(-2, -1).shape))
    print("scores:", scores, "shape:", tuple(scores.shape))
    print("weights:", weights, "shape:", tuple(weights.shape))
    print("权重行和:", weights.sum(dim=-1))
    print("output:", output, "shape:", tuple(output.shape))

    torch.testing.assert_close(scores, torch.zeros((1, 3), dtype=dtype))
    torch.testing.assert_close(weights, torch.full((1, 3), 1 / 3, dtype=dtype))
    torch.testing.assert_close(output, torch.tensor([[3.0, 2.0]], dtype=dtype))

    print("\n2. 批次例子：检查形状、权重行和与 Q/K/V 梯度")
    # 直接创建叶子 Tensor，使 backward 后可以读取各自的 .grad。
    q = torch.randn((2, 3, 2), generator=generator, dtype=dtype, requires_grad=True)
    k = torch.randn((2, 3, 2), generator=generator, dtype=dtype, requires_grad=True)
    v = torch.randn((2, 3, 2), generator=generator, dtype=dtype, requires_grad=True)

    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    output, weights = scaled_dot_product_attention(q, k, v)
    row_sums = weights.sum(dim=-1)

    print("Q / K / V shape:", tuple(q.shape), tuple(k.shape), tuple(v.shape))
    print("K.transpose(-2, -1) shape:", tuple(k.transpose(-2, -1).shape))
    print("scores shape:", tuple(scores.shape))
    print("weights shape:", tuple(weights.shape))
    print("output shape:", tuple(output.shape))
    print("权重行和:", row_sums)

    assert scores.shape == (2, 3, 3)
    assert weights.shape == (2, 3, 3)
    assert output.shape == (2, 3, 2)
    assert torch.isfinite(scores).all().item(), "scores 中出现非有限值"
    assert torch.isfinite(weights).all().item(), "weights 中出现非有限值"
    assert torch.isfinite(output).all().item(), "output 中出现非有限值"
    assert (weights >= 0).all().item(), "softmax 权重应非负"
    torch.testing.assert_close(
        row_sums, torch.ones((2, 3), dtype=dtype), rtol=0, atol=1e-12,
    )

    loss = output.square().mean()
    assert torch.isfinite(loss).item(), "loss 应为有限标量"
    loss.backward()
    print("loss = output.square().mean():", loss.item())

    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        assert tensor.grad is not None, f"{name} 没有梯度"
        assert tensor.grad.shape == tensor.shape, f"{name} 的梯度形状不匹配"
        assert torch.isfinite(tensor.grad).all().item(), f"{name} 的梯度包含非有限值"
        print(
            f"{name}.grad shape: {tuple(tensor.grad.shape)}, "
            f"最大绝对值: {tensor.grad.abs().max().item():.6f}，梯度均有限"
        )

    print("\n3. 缩放对照：观察 scores 范围和权重集中程度")
    # 各维度使用同一组随机向量的前 d 个特征；每对比较共用 Q、K、V。
    q_full = torch.randn((2, 3, 128), generator=generator, dtype=dtype)
    k_full = torch.randn((2, 3, 128), generator=generator, dtype=dtype)
    values = torch.ones((2, 3, 1), dtype=dtype)

    print("每行最大权重越接近 1，表示该行越集中在某个 key。")
    for head_dim in (2, 32, 128):
        q = q_full[..., :head_dim]
        k = k_full[..., :head_dim]
        raw_scores = q @ k.transpose(-2, -1)
        scaled_scores = raw_scores / math.sqrt(head_dim)
        raw_weights = torch.softmax(raw_scores, dim=-1)
        _, scaled_weights = scaled_dot_product_attention(q, k, values)

        print(f"head_dim = {head_dim}")
        for label, score_values, weight_values in (
            ("未缩放", raw_scores, raw_weights),
            ("已缩放", scaled_scores, scaled_weights),
        ):
            mean_max_weight = weight_values.max(dim=-1).values.mean().item()
            print(
                f"  {label}: scores 范围 "
                f"[{score_values.min().item():.4f}, {score_values.max().item():.4f}]，"
                f"每行最大权重的平均值 {mean_max_weight:.4f}"
            )

    print("缩放控制 scores 的尺度，具体权重仍由 Q 和 K 决定。")


if __name__ == "__main__":
    main()
