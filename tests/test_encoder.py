import math

import pytest
import torch

from petvision.models.transformer.blocks import (
    Encoder,
    EncoderBlock,
    FeedForward,
)
from petvision.models.transformer.positional_encoding import (
    SinusoidalPositionalEncoding,
)


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
        output[unchanged_tokens],
        perturbed_output[unchanged_tokens],
        rtol=0,
        atol=1e-12,
    )
    # 确认扰动确实改变了该 token，避免一个恒定输出也通过检查。
    assert (output[0, 1] - perturbed_output[0, 1]).abs().max().item() > 1e-6


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding", "combined"])
def test_encoderblock_forward_backward(B, mask_kind):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = (
        EncoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            dropout=0.0,
        )
        .cpu()
        .double()
    )
    assert block.addnorm1 is not block.addnorm2
    assert block.addnorm1.norm is not block.addnorm2.norm
    assert block.addnorm1.norm.gamma is not block.addnorm2.norm.gamma
    assert block.addnorm1.norm.beta is not block.addnorm2.norm.beta

    x = torch.randn(
        (B, 3, 32),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
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
        assert torch.isfinite(tensor.grad).all().item(), (
            f"{name} 梯度包含非有限值"
        )


@pytest.mark.parametrize("mask_kind", ["causal", "padding"])
def test_encoderblock_masks(mask_kind):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = (
        EncoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            dropout=0.0,
        )
        .cpu()
        .double()
        .eval()
    )
    x = torch.randn((2, 4, 32), generator=generator, dtype=torch.float64)
    perturbed = x.clone()
    # 改动位置 2、3，检查位置 0、1 是否仍然不受影响。
    perturbed[:, 2:] += 10.0
    attn_mask = None
    key_padding_mask = None
    if mask_kind in ("causal", "combined"):
        attn_mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
    if mask_kind in ("padding", "combined"):
        key_padding_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, True, True],
            ]
        )

    with torch.no_grad():
        output = block(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        perturbed_output = block(
            perturbed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        # 去掉 Mask 后应能观察到扰动，确认输入确实能影响这些 query。
        unmasked_output = block(x)
        unmasked_perturbed_output = block(perturbed)

    torch.testing.assert_close(
        output[:, :2],
        perturbed_output[:, :2],
        rtol=0,
        atol=1e-12,
    )
    unmasked_difference = (
        (unmasked_output[:, :2] - unmasked_perturbed_output[:, :2])
        .abs()
        .max()
        .item()
    )
    assert unmasked_difference > 1e-6
    # padding query 仍有自己的输入和残差，不要求它的输出不变或为零。


@pytest.mark.parametrize("checked_mask", ["causal", "padding"])
def test_encoderblock_combined_masks(checked_mask):
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = (
        EncoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            dropout=0.0,
        )
        .cpu()
        .double()
        .eval()
    )
    x = torch.randn((2, 4, 32), generator=generator, dtype=torch.float64)
    attn_mask = torch.ones((4, 4), dtype=torch.bool).triu(1)
    # 本例专门在中间屏蔽一个 key，使后面的有效 query 能单独检验 padding。
    # 训练仍可使用右侧 padding；Mask 接口也允许下面这样的布尔表。
    key_padding_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
        ]
    )
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
        output = block(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        perturbed_output = block(
            perturbed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        # 只去掉正在检查的 Mask，保留另一个，确认二者不能互相替代。
        control_output = block(
            x,
            attn_mask=control_attn_mask,
            key_padding_mask=control_key_padding_mask,
        )
        control_perturbed_output = block(
            perturbed,
            attn_mask=control_attn_mask,
            key_padding_mask=control_key_padding_mask,
        )

    torch.testing.assert_close(
        output[:, query_index],
        perturbed_output[:, query_index],
        rtol=0,
        atol=1e-12,
    )
    control_difference = (
        (
            control_output[:, query_index]
            - control_perturbed_output[:, query_index]
        )
        .abs()
        .amax(dim=-1)
    )
    assert (control_difference > 1e-6).all().item()


def test_encoderblock_train_eval():
    torch.manual_seed(42)
    generator = torch.Generator(device="cpu").manual_seed(42)
    block = (
        EncoderBlock(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            dropout=0.1,
        )
        .cpu()
        .double()
    )
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


def test_positional_encoding_values():
    position = SinusoidalPositionalEncoding(embed_dim=4, dropout=0.0).cpu()
    x = torch.zeros((1, 3, 4))
    output = position(x)

    # C=4 时，两对通道的分母分别为 1、100。
    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)],
        ]
    )
    assert output.shape == x.shape
    torch.testing.assert_close(output[0, :2], expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(position(x + 1), output + 1, rtol=0, atol=1e-7)
    assert not torch.equal(output[:, 0], output[:, 1])

    # 固定位置表跟随模块保存、迁移，但不作为参数交给优化器。
    assert dict(position.named_buffers())["pe"] is position.pe
    assert "pe" not in dict(position.named_parameters())
    assert not position.pe.requires_grad
    print("位置 0、1 的编码:", output[0, :2].tolist())
    print("buffer:", list(dict(position.named_buffers())))


def test_encoder_stack():
    torch.manual_seed(42)
    encoder = (
        Encoder(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            depth=2,
            dropout=0.0,
        )
        .cpu()
        .double()
    )
    x = torch.randn((2, 8, 32), dtype=torch.float64, requires_grad=True)
    assert len(encoder.blocks) == 2
    assert encoder.blocks[0] is not encoder.blocks[1]

    output = encoder(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all().item()

    # 手动走一遍同一条路径，观察入口与两层的形状；不建立第二套模型。
    with torch.no_grad():
        stage = encoder.position(x * math.sqrt(encoder.embed_dim))
        print("入口缩放、位置相加后:", tuple(stage.shape))
        for index, block in enumerate(encoder.blocks, start=1):
            stage = block(stage)
            assert stage.shape == x.shape
            print(f"Block {index} 输出:", tuple(stage.shape))
        torch.testing.assert_close(output, stage, rtol=0, atol=1e-12)

    # 随机 probe 避免 LayerNorm 输出直接求和导致的退化梯度。
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    representatives = [
        ("输入", x),
        ("第一层 Q 投影", encoder.blocks[0].attention.q_proj.weight),
        ("最后一层 FFN 输出投影", encoder.blocks[-1].ffn.dense2.weight),
    ]
    for name, tensor in representatives:
        assert tensor.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(tensor.grad).all().item(), (
            f"{name} 梯度包含非有限值"
        )
        print(f"{name} 梯度:", tuple(tensor.grad.shape), "有限")


def test_encoder_causal_mask():
    torch.manual_seed(42)
    encoder = (
        Encoder(
            embed_dim=32,
            num_heads=4,
            ffn_hidden_dim=128,
            depth=2,
            dropout=0.0,
        )
        .cpu()
        .double()
        .eval()
    )
    x = torch.randn((2, 8, 32), dtype=torch.float64)
    perturbed = x.clone()
    perturbed[:, 4:] += 10.0
    attn_mask = torch.ones((8, 8), dtype=torch.bool).triu(1)

    with torch.no_grad():
        output = encoder(x, attn_mask=attn_mask)
        perturbed_output = encoder(perturbed, attn_mask=attn_mask)
        # 同一模型去掉 Mask，观察双向 Attention 能读取后缀的区别。
        unmasked_output = encoder(x)
        unmasked_perturbed_output = encoder(perturbed)

    torch.testing.assert_close(
        output[:, :4],
        perturbed_output[:, :4],
        rtol=0,
        atol=1e-6,
    )
    masked_difference = (
        (output[:, :4] - perturbed_output[:, :4]).abs().max().item()
    )
    unmasked_difference = (
        (unmasked_output[:, :4] - unmasked_perturbed_output[:, :4])
        .abs()
        .max()
        .item()
    )
    assert unmasked_difference > 1e-6
    print(
        f"后缀扰动后的前缀差: causal={masked_difference:.6e}, "
        f"双向={unmasked_difference:.6e}"
    )
