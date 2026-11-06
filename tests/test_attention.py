import pytest
import torch

from petvision.models.transformer.attention import MultiHeadSelfAttention


def test_attention_shapes_and_gradients():
    generator = torch.Generator(device="cpu").manual_seed(42)

    for B in (1, 2):
        for T in (1, 8):
            # 每组新建模型和叶子输入，避免上一组的梯度累加到本组。
            multi = MultiHeadSelfAttention(
                embed_dim=32, num_heads=4, dropout=0.0, bias=False,
            ).cpu().double()
            # 后续参考层的权重映射使用这些名称，四个投影各有独立参数。
            projections = [multi.q_proj, multi.k_proj, multi.v_proj, multi.out_proj]
            assert len({id(layer) for layer in projections}) == 4
            x = torch.randn(
                (B, T, 32), generator=generator,
                dtype=torch.float64, requires_grad=True,
            )

            # 默认调用返回 Tensor；要求返回权重时，先 output、后 weights。
            output = multi(x)
            assert isinstance(output, torch.Tensor)
            assert output.shape == (B, T, 32)
            assert output.dtype == x.dtype
            assert output.device == x.device
            assert torch.isfinite(output).all().item(), "输出包含非有限值"

            result = multi(x, return_weights=True)
            assert isinstance(result, tuple) and len(result) == 2
            output_with_weights, weights = result
            assert output_with_weights.shape == (B, T, 32)
            assert weights.shape == (B, 4, T, T)
            assert torch.isfinite(weights).all().item(), "权重包含非有限值"
            assert (weights >= 0).all().item(), "softmax 权重应非负"

            # dropout=0，两种返回方式得到的输出应一致，每个 query 行和为 1。
            torch.testing.assert_close(
                output_with_weights, output, rtol=0, atol=1e-12,
            )
            torch.testing.assert_close(
                weights.sum(dim=-1), torch.ones((B, 4, T), dtype=x.dtype),
                rtol=0, atol=1e-12,
            )

            loss = output.square().mean()
            assert loss.ndim == 0
            assert torch.isfinite(loss).item(), "loss 应为有限标量"
            loss.backward()

            assert x.grad is not None, "输入 x 没有梯度"
            assert x.grad.shape == x.shape
            assert torch.isfinite(x.grad).all().item(), "输入梯度包含非有限值"

            # T=1 时 Q/K 梯度可以为零；这里检查存在、形状正确且有限。
            for name, parameter in multi.named_parameters():
                assert parameter.grad is not None, f"{name} 没有梯度"
                assert parameter.grad.shape == parameter.shape, f"{name} 梯度形状错误"
                assert torch.isfinite(parameter.grad).all().item(), (
                    f"{name} 梯度包含非有限值"
                )

            print(
                f"B={B}, T={T}, output={tuple(output.shape)}, "
                f"weights={tuple(weights.shape)}, loss={loss.item():.6f}"
            )


@pytest.mark.parametrize("embed_dim,num_heads", [
    (8, 3),
    (0, 2),
    (-8, 2),
    (8.0, 2),
    (True, 1),
    (8, 0),
    (8, -2),
    (8, 2.0),
    (8, True),
])
def test_attention_invalid_heads(embed_dim, num_heads):
    # 在构造时明确报 ValueError，而非推迟到整除或 reshape 时才出错。
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)


@pytest.mark.parametrize("mask_name", ["attn_mask", "key_padding_mask"])
@pytest.mark.parametrize("B,T", [(2, 3), (1, 1)])
def test_attention_rejects_unimplemented_masks(mask_name, B, T, monkeypatch):
    multi = MultiHeadSelfAttention(8, 2)
    x = torch.randn(B, T, 8)
    mask_shape = (T, T) if mask_name == "attn_mask" else (B, T)
    mask = torch.zeros(mask_shape, dtype=torch.bool)

    def unexpected_projection(*args, **kwargs):
        pytest.fail("传入未支持的 Mask 后不应继续执行投影")

    monkeypatch.setattr(multi.q_proj, "forward", unexpected_projection)

    # 全 False 的 Mask 仍然是非 None；多元素和单元素 Tensor 均应拒绝。
    with pytest.raises(NotImplementedError, match="Mask 在 Day 3 接入"):
        multi(x, **{mask_name: mask})


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("B,T", [(1, 1), (1, 8), (2, 1), (2, 8)])
def test_attention_reference(bias, B, T):
    # 全局种子固定模型初始化；独立生成器固定输入与两边共用的 probe。
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    embed_dim = 32
    tolerance = 1e-6

    attention = MultiHeadSelfAttention(
        embed_dim=embed_dim, num_heads=4, dropout=0.0, bias=bias,
    ).cpu().double().eval()
    reference = torch.nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=4, dropout=0.0,
        bias=bias, batch_first=True,
    ).cpu().double().eval()

    # 参考层将 Q/K/V 的参数合在一起，按行依次存放三块 [C,C]。
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([
            attention.q_proj.weight,
            attention.k_proj.weight,
            attention.v_proj.weight,
        ], dim=0))
        reference.out_proj.weight.copy_(attention.out_proj.weight)
        if bias:
            reference.in_proj_bias.copy_(torch.cat([
                attention.q_proj.bias,
                attention.k_proj.bias,
                attention.v_proj.bias,
            ], dim=0))
            reference.out_proj.bias.copy_(attention.out_proj.bias)

    # 数值相同但相互独立的叶子输入，分别保存两次 backward 的梯度。
    x = torch.randn((B, T, embed_dim), generator=generator, dtype=torch.float64)
    x_custom = x.detach().clone().requires_grad_(True)
    x_reference = x.detach().clone().requires_grad_(True)
    attention.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)

    output, weights = attention(x_custom, return_weights=True)
    reference_output, reference_weights = reference(
        x_reference, x_reference, x_reference,
        need_weights=True, average_attn_weights=False,
    )
    assert output.shape == reference_output.shape == (B, T, embed_dim)
    assert weights.shape == reference_weights.shape == (B, 4, T, T)

    # 同一个非平凡标量目标；eval 切换层行为，仍然允许求导。
    probe = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    (output * probe).sum().backward()
    (reference_output * probe).sum().backward()

    for label, model in (("自写层", attention), ("参考层", reference)):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{label} {name} 没有梯度"
            assert parameter.grad.shape == parameter.shape
            assert torch.isfinite(parameter.grad).all().item(), (
                f"{label} {name} 梯度包含非有限值"
            )
    assert x_custom.grad is not None, "自写层输入没有梯度"
    assert x_reference.grad is not None, "参考层输入没有梯度"

    comparisons = {
        "output": (output, reference_output),
        "weights": (weights, reference_weights),
        "input.grad": (x_custom.grad, x_reference.grad),
        "out_proj.weight.grad": (
            attention.out_proj.weight.grad, reference.out_proj.weight.grad,
        ),
    }

    # 将参考层的合并梯度重新分成 Q、K、V 三块，逐项比较。
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        projection = getattr(attention, name)
        rows = slice(index * embed_dim, (index + 1) * embed_dim)
        comparisons[f"{name}.weight.grad"] = (
            projection.weight.grad, reference.in_proj_weight.grad[rows],
        )
        if bias:
            comparisons[f"{name}.bias.grad"] = (
                projection.bias.grad, reference.in_proj_bias.grad[rows],
            )
    if bias:
        comparisons["out_proj.bias.grad"] = (
            attention.out_proj.bias.grad, reference.out_proj.bias.grad,
        )

    for name, (actual, expected) in comparisons.items():
        assert actual.shape == expected.shape, f"{name} 形状不一致"
        assert torch.isfinite(actual).all().item(), f"{name} 包含非有限值"
        assert torch.isfinite(expected).all().item(), f"参考 {name} 包含非有限值"
        max_error = (actual - expected).abs().max().item()
        print(
            f"B={B}, T={T}, bias={bias}, {name}: shape={tuple(actual.shape)}, "
            f"max_abs_error={max_error:.6e}, atol={tolerance:.0e}"
        )
        assert max_error < tolerance, f"{name} 最大绝对误差 {max_error} 超过容差"
