import math

import pytest
import torch

from petvision.models.transformer import normalization


def test_layernorm_known_values():
    eps = 1e-5
    layer = normalization.LayerNorm(embed_dim=2, eps=eps).cpu().double()
    x = torch.tensor([[[1.0, 3.0]]], dtype=torch.float64)

    # 均值为 2、方差为 1；期望值也包含 eps，不能直接用精确的 [-1,1]。
    expected = torch.tensor([[[-1.0, 1.0]]], dtype=x.dtype) / math.sqrt(1 + eps)
    output = layer(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype and output.device == x.device
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-12)

    # 只改一个 token 的一个特征，其他 token 的输出应逐项不变。
    tokens = torch.tensor([
        [[1.0, 3.0], [2.0, 4.0], [5.0, 9.0]],
        [[0.0, 2.0], [3.0, 7.0], [4.0, 8.0]],
    ], dtype=x.dtype)
    perturbed = tokens.clone()
    perturbed[0, 1, 0] += 5.0
    original_output = layer(tokens)
    perturbed_output = layer(perturbed)
    unchanged_tokens = torch.ones((2, 3), dtype=torch.bool)
    unchanged_tokens[0, 1] = False
    torch.testing.assert_close(
        original_output[unchanged_tokens], perturbed_output[unchanged_tokens],
        rtol=0, atol=1e-12,
    )
    assert (original_output[0, 1] - perturbed_output[0, 1]).abs().max().item() > 1e-6
    print("手算 [1,3] 的输出:", output.detach().tolist())


@pytest.mark.parametrize("embed_dim", [1, 2, 32])
def test_layernorm_constant_input(embed_dim):
    layer = normalization.LayerNorm(embed_dim=embed_dim).cpu().double()
    x = torch.full((2, 3, embed_dim), 7.0, dtype=torch.float64, requires_grad=True)
    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, torch.zeros_like(x), rtol=0, atol=1e-12)

    # C=1 的方差也应为零；eps 使归一化和反向都保持有限。
    generator = torch.Generator(device="cpu").manual_seed(42)
    probe = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    (output * probe).sum().backward()
    for name, tensor in (("x", x), ("gamma", layer.gamma), ("beta", layer.beta)):
        assert tensor.grad is not None, f"{name} 没有梯度"
        assert tensor.grad.shape == tensor.shape
        assert torch.isfinite(tensor.grad).all().item(), f"{name} 梯度包含非有限值"


@pytest.mark.parametrize("embed_dim,input_shape", [
    (32, (2, 3, 1)),
    (1, (2, 3, 32)),
    (32, (2, 3, 16)),
    (32, ()),
])
def test_layernorm_rejects_invalid_shape(embed_dim, input_shape):
    layer = normalization.LayerNorm(embed_dim).cpu().double()
    x = torch.zeros(input_shape, dtype=torch.float64)

    # 前两种错误能被广播，因此必须在计算前明确拒绝。
    with pytest.raises(ValueError, match="embed_dim"):
        layer(x)


@pytest.mark.parametrize("B,T", [(1, 1), (1, 3), (2, 1), (2, 3)])
def test_layernorm_reference(B, T):
    generator = torch.Generator(device="cpu").manual_seed(42)
    embed_dim = 32
    eps = 1e-5
    tolerance = 1e-6
    layer = normalization.LayerNorm(embed_dim=embed_dim, eps=eps).cpu().double()
    reference = torch.nn.LayerNorm(embed_dim, eps=eps).cpu().double()

    # 用非默认的缩放与偏移，检查两个参数都实际参与了计算。
    with torch.no_grad():
        layer.gamma.copy_(0.5 + torch.rand(embed_dim, generator=generator, dtype=torch.float64))
        layer.beta.copy_(torch.randn(embed_dim, generator=generator, dtype=torch.float64))
        reference.weight.copy_(layer.gamma)
        reference.bias.copy_(layer.beta)

    # 同一份数值，各自独立的叶子输入，分别保存 backward 的梯度。
    x = torch.randn((B, T, embed_dim), generator=generator, dtype=torch.float64)
    x_custom = x.detach().clone().requires_grad_(True)
    x_reference = x.detach().clone().requires_grad_(True)
    output = layer(x_custom)
    reference_output = reference(x_reference)
    assert output.shape == reference_output.shape == x.shape

    # 两边使用相同 probe，避免归一化输出直接求和带来的退化例子。
    probe = torch.randn(output.shape, generator=generator, dtype=x.dtype)
    (output * probe).sum().backward()
    (reference_output * probe).sum().backward()

    comparisons = {
        "output": (output, reference_output),
        "input.grad": (x_custom.grad, x_reference.grad),
        "gamma/weight.grad": (layer.gamma.grad, reference.weight.grad),
        "beta/bias.grad": (layer.beta.grad, reference.bias.grad),
    }
    for name, (actual, expected) in comparisons.items():
        assert actual is not None and expected is not None, f"{name} 缺少结果或梯度"
        assert actual.shape == expected.shape, f"{name} 形状不一致"
        assert torch.isfinite(actual).all().item(), f"{name} 包含非有限值"
        assert torch.isfinite(expected).all().item(), f"参考 {name} 包含非有限值"
        max_error = (actual - expected).abs().max().item()
        print(
            f"B={B}, T={T}, {name}: shape={tuple(actual.shape)}, "
            f"max_abs_error={max_error:.6e}, atol={tolerance:.0e}"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
