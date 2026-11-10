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
def test_attention_all_false_masks(mask_name, B, T):
    multi = MultiHeadSelfAttention(8, 2)
    x = torch.randn(B, T, 8)
    mask_shape = (T, T) if mask_name == "attn_mask" else (B, T)
    mask = torch.zeros(mask_shape, dtype=torch.bool)

    # 单元素和多元素的全 False 表都不应改变未屏蔽的结果。
    torch.testing.assert_close(multi(x, **{mask_name: mask}), multi(x))


@pytest.mark.parametrize("mask_name", ["attn_mask", "key_padding_mask"])
@pytest.mark.parametrize("invalid_kind,error_type", [
    ("not_tensor", TypeError),
    ("dtype", TypeError),
    ("shape", ValueError),
    ("device", ValueError),
])
def test_attention_invalid_masks(mask_name, invalid_kind, error_type):
    multi = MultiHeadSelfAttention(8, 2)
    x = torch.randn(2, 4, 8)
    mask_shape = (4, 4) if mask_name == "attn_mask" else (2, 4)
    mask = torch.zeros(mask_shape, dtype=torch.bool)

    if invalid_kind == "not_tensor":
        mask = mask.tolist()
    elif invalid_kind == "dtype":
        mask = mask.float()
    elif invalid_kind == "shape":
        # [1,T] 可以广播，却不符合项目约定的 Mask 形状。
        mask = torch.zeros((1, 4), dtype=torch.bool)
    elif invalid_kind == "device":
        # meta 无需 GPU，用它验证不同设备的 Mask 会被提前拒绝。
        mask = torch.empty(mask_shape, dtype=torch.bool, device="meta")

    with pytest.raises(error_type, match=mask_name):
        multi(x, **{mask_name: mask})


def test_attention_causal_known_values():
    multi = MultiHeadSelfAttention(8, 2, dropout=0.0).double().eval()
    with torch.no_grad():
        # Q/K 为零，所有分数都为零；V 和输出投影保留输入数值。
        multi.q_proj.weight.zero_()
        multi.k_proj.weight.zero_()
        multi.v_proj.weight.copy_(torch.eye(8, dtype=torch.float64))
        multi.out_proj.weight.copy_(torch.eye(8, dtype=torch.float64))

    # 四个 token 的八个特征分别全为 1、3、5、7，方便逐行手算。
    values = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    x = values.reshape(1, 4, 1).expand(1, 4, 8)
    causal_mask = torch.ones((4, 4), dtype=torch.bool).triu(diagonal=1)
    output, weights = multi(x, attn_mask=causal_mask, return_weights=True)

    expected_weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1 / 2, 1 / 2, 0.0, 0.0],
        [1 / 3, 1 / 3, 1 / 3, 0.0],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ], dtype=torch.float64)
    expected_values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        weights, expected_weights[None, None, :, :].expand(1, 2, 4, 4),
        rtol=0, atol=1e-12,
    )
    torch.testing.assert_close(
        output, expected_values.reshape(1, 4, 1).expand_as(output),
        rtol=0, atol=1e-12,
    )

    print("causal mask（1 表示屏蔽）:\n", causal_mask.int())
    print("第一个头的权重:\n", weights[0, 0].detach())
    print("每个 token 的输出特征值:", output[0, :, 0].detach())


def test_attention_causal_mask():
    torch.manual_seed(42)
    multi = MultiHeadSelfAttention(8, 2, dropout=0.0).double().eval()
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    causal_mask = torch.ones((4, 4), dtype=torch.bool).triu(diagonal=1)
    tolerance = 1e-6

    output, weights = multi(x, attn_mask=causal_mask, return_weights=True)
    perturbed = x.clone()
    perturbed[:, 2:] += 10.0
    perturbed_output = multi(perturbed, attn_mask=causal_mask)

    # 改动未来的位置 2、3，不应影响位置 0、1 的输出。
    prefix_error = (output[:, :2] - perturbed_output[:, :2]).abs().max().item()
    torch.testing.assert_close(
        output[:, :2], perturbed_output[:, :2], rtol=0, atol=tolerance,
    )
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(perturbed_output).all().item()
    blocked = causal_mask[None, None, :, :].expand_as(weights)
    assert (weights.masked_select(blocked) == 0).all().item()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones((2, 2, 4), dtype=x.dtype),
        rtol=0, atol=1e-12,
    )

    # 去掉 Mask 后，同样的未来扰动应能影响前缀，确认例子能检出信息泄漏。
    unmasked_output = multi(x)
    unmasked_perturbed_output = multi(perturbed)
    unmasked_error = (
        unmasked_output[:, :2] - unmasked_perturbed_output[:, :2]
    ).abs().max().item()
    assert unmasked_error > tolerance
    print(
        f"causal: prefix_shape={tuple(output[:, :2].shape)}, "
        f"max_abs_error={prefix_error:.6e}, atol={tolerance:.0e}, "
        f"without_mask_error={unmasked_error:.6e}"
    )


def test_attention_padding_mask():
    torch.manual_seed(42)
    multi = MultiHeadSelfAttention(8, 2, dropout=0.0).double().eval()
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    padding_mask = torch.tensor([
        [False, False, True, True],
        [False, False, False, True],
    ])
    tolerance = 1e-6

    output, weights = multi(x, key_padding_mask=padding_mask, return_weights=True)
    perturbed = x.clone()
    perturbed[padding_mask] += 10.0
    perturbed_output = multi(perturbed, key_padding_mask=padding_mask)

    assert torch.isfinite(output).all().item()
    assert torch.isfinite(perturbed_output).all().item()
    blocked = padding_mask[:, None, None, :].expand_as(weights)
    assert (weights.masked_select(blocked) == 0).all().item()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones((2, 2, 4), dtype=x.dtype),
        rtol=0, atol=1e-12,
    )

    unmasked_output = multi(x)
    unmasked_perturbed_output = multi(perturbed)
    # 只改 PAD，逐条比较有效 query；PAD query 的输出不要求相同或为零。
    for batch_index, valid_length in enumerate((2, 3)):
        valid_output = output[batch_index, :valid_length]
        perturbed_valid_output = perturbed_output[batch_index, :valid_length]
        max_error = (valid_output - perturbed_valid_output).abs().max().item()
        torch.testing.assert_close(
            valid_output, perturbed_valid_output, rtol=0, atol=tolerance,
        )
        # 同一份输入在不使用 Mask 时，应能观察到 PAD 扰动的影响。
        unmasked_error = (
            unmasked_output[batch_index, :valid_length]
            - unmasked_perturbed_output[batch_index, :valid_length]
        ).abs().max().item()
        assert unmasked_error > tolerance
        print(
            f"padding: batch={batch_index}, valid_shape={tuple(valid_output.shape)}, "
            f"max_abs_error={max_error:.6e}, atol={tolerance:.0e}, "
            f"without_mask_error={unmasked_error:.6e}"
        )


def test_attention_combined_masks():
    torch.manual_seed(42)
    multi = MultiHeadSelfAttention(8, 2, dropout=0.0).double().eval()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    causal_mask = torch.ones((4, 4), dtype=torch.bool).triu(diagonal=1)
    padding_mask = torch.tensor([
        [False, False, True, True],
        [False, False, False, True],
    ])

    output, weights = multi(
        x, attn_mask=causal_mask, key_padding_mask=padding_mask, return_weights=True,
    )
    blocked = (
        causal_mask[None, None, :, :] | padding_mask[:, None, None, :]
    ).expand_as(weights)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(weights).all().item()
    assert (weights.masked_select(blocked) == 0).all().item()
    # OR 后仍允许的 key 都应保留，防止误将其他位置也屏蔽。
    assert (weights.masked_select(~blocked) > 0).all().item()
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones((2, 2, 4), dtype=x.dtype),
        rtol=0, atol=1e-12,
    )

    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    for name, parameter in multi.named_parameters():
        assert parameter.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度包含非有限值"
    masked_weight_max = weights.masked_select(blocked).abs().max().item()
    row_sum_error = (weights.sum(dim=-1) - 1).abs().max().item()
    print(
        f"combined: weights_shape={tuple(weights.shape)}, "
        f"masked_weight_max={masked_weight_max:.6e}, "
        f"row_sum_max_abs_error={row_sum_error:.6e}, atol=1e-12, "
        "output_and_gradients_finite=True"
    )


@pytest.mark.parametrize("mask_kind", ["attn", "padding", "combined"])
def test_attention_rejects_fully_masked_query(mask_kind):
    multi = MultiHeadSelfAttention(8, 2)
    x = torch.randn(2, 4, 8)
    attn_mask = None
    padding_mask = None

    if mask_kind == "attn":
        attn_mask = torch.zeros((4, 4), dtype=torch.bool)
        attn_mask[1, :] = True
    elif mask_kind == "padding":
        padding_mask = torch.zeros((2, 4), dtype=torch.bool)
        padding_mask[0, :] = True
    else:
        attn_mask = torch.ones((4, 4), dtype=torch.bool).triu(diagonal=1)
        padding_mask = torch.zeros((2, 4), dtype=torch.bool)
        padding_mask[0, 0] = True
        # 两张表单独都合法，合并后第 0 条序列的 query 0 没有可读取的 key。

    with pytest.raises(ValueError, match="没有可关注 key") as error:
        multi(x, attn_mask=attn_mask, key_padding_mask=padding_mask)
    print(f"fully_masked: kind={mask_kind}, ValueError={error.value}")


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
