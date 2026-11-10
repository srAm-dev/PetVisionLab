import pytest
import torch

from petvision.models.transformer.blocks import EncoderBlock, FeedForward


def test_feedforward_token_independence():
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    ffn = FeedForward(embed_dim=32, hidden_dim=128).cpu().double().eval()
    x = torch.randn((2, 3, 32), generator=generator, dtype=torch.float64)
    perturbed = x.clone()
    perturbed[0, 1] += 10.0

    with torch.no_grad():
        output = ffn(x)
        perturbed_output = ffn(perturbed)

    assert output.shape == perturbed_output.shape == x.shape
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(perturbed_output).all().item()
    unchanged_tokens = torch.ones((2, 3), dtype=torch.bool)
    unchanged_tokens[0, 1] = False
    torch.testing.assert_close(
        output[unchanged_tokens], perturbed_output[unchanged_tokens],
        rtol=0, atol=1e-12,
    )
    # 确认扰动确实改变了该 token，避免一个恒定输出也通过检查。
    assert (output[0, 1] - perturbed_output[0, 1]).abs().max().item() > 1e-6


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding", "combined"])
def test_encoderblock_forward_backward(B, mask_kind):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = EncoderBlock(
        embed_dim=32, num_heads=4, ffn_hidden_dim=128, dropout=0.0,
    ).cpu().double()
    assert block.addnorm1 is not block.addnorm2
    assert block.addnorm1.norm is not block.addnorm2.norm
    assert block.addnorm1.norm.gamma is not block.addnorm2.norm.gamma
    assert block.addnorm1.norm.beta is not block.addnorm2.norm.beta

    x = torch.randn(
        (B, 3, 32), generator=generator, dtype=torch.float64, requires_grad=True,
    )
    attn_mask = None
    key_padding_mask = None
    if mask_kind in ("causal", "combined"):
        attn_mask = torch.ones((3, 3), dtype=torch.bool).triu(1)
    if mask_kind in ("padding", "combined"):
        key_padding_mask = torch.zeros((B, 3), dtype=torch.bool)
        key_padding_mask[:, -1] = True
        if B == 2:
            key_padding_mask[1, -2:] = True

    output = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype and output.device == x.device
    assert torch.isfinite(output).all().item()

    # LayerNorm 的输出直接求和容易退化，使用随机 probe 检查反向路径。
    probe = torch.randn(output.shape, generator=generator, dtype=output.dtype)
    (output * probe).sum().backward()
    for name, tensor in [("input", x), *block.named_parameters()]:
        assert tensor.grad is not None, f"{name} 没有梯度"
        assert tensor.grad.shape == tensor.shape, f"{name} 梯度形状错误"
        assert torch.isfinite(tensor.grad).all().item(), f"{name} 梯度包含非有限值"


@pytest.mark.parametrize("mask_kind", ["causal", "padding"])
def test_encoderblock_masks(mask_kind):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = EncoderBlock(
        embed_dim=32, num_heads=4, ffn_hidden_dim=128, dropout=0.0,
    ).cpu().double().eval()
    x = torch.randn((2, 4, 32), generator=generator, dtype=torch.float64)
    perturbed = x.clone()
    # 改动位置 2、3，检查位置 0、1 是否仍然不受影响。
    perturbed[:, 2:] += 10.0
    attn_mask = None
    key_padding_mask = None
    if mask_kind in ("causal", "combined"):
        attn_mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
    if mask_kind in ("padding", "combined"):
        key_padding_mask = torch.tensor([
            [False, False, True, True],
            [False, False, True, True],
        ])

    with torch.no_grad():
        output = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        perturbed_output = block(
            perturbed, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
        )
        # 去掉 Mask 后应能观察到扰动，确认输入确实能影响这些 query。
        unmasked_output = block(x)
        unmasked_perturbed_output = block(perturbed)

    torch.testing.assert_close(
        output[:, :2], perturbed_output[:, :2], rtol=0, atol=1e-12,
    )
    unmasked_difference = (
        unmasked_output[:, :2] - unmasked_perturbed_output[:, :2]
    ).abs().max().item()
    assert unmasked_difference > 1e-6
    # padding query 仍有自己的输入和残差，不要求它的输出不变或为零。


@pytest.mark.parametrize("checked_mask", ["causal", "padding"])
def test_encoderblock_combined_masks(checked_mask):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = EncoderBlock(
        embed_dim=32, num_heads=4, ffn_hidden_dim=128, dropout=0.0,
    ).cpu().double().eval()
    x = torch.randn((2, 4, 32), generator=generator, dtype=torch.float64)
    attn_mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
    # 本例专门在中间屏蔽一个 key，使后面的有效 query 能单独检验 padding。
    # 训练仍可使用右侧 padding；Mask 接口也允许下面这样的布尔表。
    key_padding_mask = torch.tensor([
        [False, True, False, False],
        [False, False, True, False],
    ])
    perturbed = x.clone()
    if checked_mask == "causal":
        # 位置 3 是有效的未来 key，padding 无法代替 causal 屏蔽它。
        perturbed[:, 3] += 10.0
        query_index = 0
        control_attn_mask = None
        control_key_padding_mask = key_padding_mask
    else:
        # 对位置 3 来说，这些 key 在过去，causal 无法代替 padding 屏蔽它们。
        perturbed[key_padding_mask] += 10.0
        query_index = 3
        control_attn_mask = attn_mask
        control_key_padding_mask = None

    with torch.no_grad():
        output = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        perturbed_output = block(
            perturbed, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
        )
        # 只去掉正在检查的 Mask，保留另一个，确认二者不能互相替代。
        control_output = block(
            x, attn_mask=control_attn_mask, key_padding_mask=control_key_padding_mask,
        )
        control_perturbed_output = block(
            perturbed, attn_mask=control_attn_mask,
            key_padding_mask=control_key_padding_mask,
        )

    torch.testing.assert_close(
        output[:, query_index], perturbed_output[:, query_index],
        rtol=0, atol=1e-12,
    )
    control_difference = (
        control_output[:, query_index] - control_perturbed_output[:, query_index]
    ).abs().amax(dim=-1)
    assert (control_difference > 1e-6).all().item()


def test_encoderblock_train_eval():
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = EncoderBlock(
        embed_dim=32, num_heads=4, ffn_hidden_dim=128, dropout=0.1,
    ).cpu().double()
    x = torch.randn((2, 3, 32), generator=generator, dtype=torch.float64)
    dropout_layers = [
        block.attention.dropout,
        block.addnorm1.dropout,
        block.addnorm2.dropout,
    ]
    assert all(layer.p == 0.1 for layer in dropout_layers)

    block.train()
    assert all(module.training for module in block.modules())
    with torch.no_grad():
        train_output = block(x)
        second_train_output = block(x)
    assert torch.isfinite(train_output).all().item()
    assert torch.isfinite(second_train_output).all().item()
    assert not torch.equal(train_output, second_train_output)

    block.eval()
    assert all(not module.training for module in block.modules())
    with torch.no_grad():
        eval_output = block(x)
        second_eval_output = block(x)
    assert eval_output.shape == x.shape
    assert torch.isfinite(eval_output).all().item()
    torch.testing.assert_close(eval_output, second_eval_output, rtol=0, atol=0)

    block.train()
    assert all(module.training for module in block.modules())
