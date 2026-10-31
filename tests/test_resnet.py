import torch
from petvision.models.cv.cnn import CNN
from petvision.models.cv.resnet import BasicBlock, ResNet18

def test_cnn_forward_backward():
    for batch_size in (1, 4):
        for bn in (True, False):
            # 每次新建模型，分别检查单张图片和一个小批次。
            model = CNN(use_bn=bn).cpu()
            images = torch.randn(batch_size, 3, 32, 32)
            labels = torch.randint(0, 10, (batch_size,), dtype=torch.long)

            # 前向：每张图片得到 10 个类别的 logits。
            model.zero_grad()
            logits = model(images)
            assert logits.shape == (batch_size, 10)

            # 损失：将 logits 与正确类别编号比较，得到一个有限标量。
            loss = torch.nn.functional.cross_entropy(logits, labels)
            assert loss.ndim == 0
            assert torch.isfinite(loss).item()

            # 反向：为参与计算的模型参数计算梯度。
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, f"{name} 没有梯度"
                assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"

            print(f"batch={batch_size}, use_bn={bn}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")

def test_cnn_stage_shapes():
    model = CNN().cpu().eval()
    images = torch.randn(2, 3, 32, 32)

    # 键是 model.net 中的层号，从 0 开始。
    expected_shapes = {
        3: (2, 32, 16, 16),  # 第一次池化后
        7: (2, 64, 8, 8),    # 第二次池化后
        10: (2, 128, 8, 8),  # 第三段卷积、BN、ReLU 后
        11: (2, 128, 1, 1),  # 全局平均池化后
        12: (2, 128),        # Flatten 后
        13: (2, 10),         # 分类层后
    }

    with torch.no_grad():
        for index, layer in enumerate(model.net):
            images = layer(images)
            shape = tuple(images.shape)

            print(f"{index}: {type(layer).__name__} -> {shape}")

            if index in expected_shapes:
                assert shape == expected_shapes[index], (
                    f"第 {index} 层形状错误："
                    f"预期 {expected_shapes[index]}，实际 {shape}"
                )

def test_batchnorm_train_eval():
    bn = torch.nn.BatchNorm2d(2).cpu()
    images = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4)

    # 1. 保存训练前的统计；clone 避免快照随原 buffer 一起改变。
    mean_before = bn.running_mean.clone()
    var_before = bn.running_var.clone()
    batches_before = bn.num_batches_tracked.clone()

    # 2. no_grad 关闭梯度记录，但 train 模式仍会更新运行统计。
    bn.train()
    with torch.no_grad():
        train_output = bn(images)

    assert train_output.shape == images.shape
    assert torch.isfinite(train_output).all().item()
    assert bn.num_batches_tracked.item() == batches_before.item() + 1
    assert not torch.equal(bn.running_mean, mean_before)
    assert not torch.equal(bn.running_var, var_before)
    batch_mean = images.mean(dim=(0, 2, 3))
    expected_mean = (1 - bn.momentum) * mean_before + bn.momentum * batch_mean
    torch.testing.assert_close(bn.running_mean, expected_mean)

    # 3. 保存训练后的统计，再检查 eval 前向不会继续更新它们。
    mean_after = bn.running_mean.clone()
    var_after = bn.running_var.clone()
    batches_after = bn.num_batches_tracked.clone()
    bn.eval()
    with torch.no_grad():
        eval_output_1 = bn(images)
        eval_output_2 = bn(images)

        # eval 使用保存的运行统计，再应用可学习的缩放和偏移。
        expected_eval = (
            (images - mean_after.view(1, 2, 1, 1))
            / torch.sqrt(var_after.view(1, 2, 1, 1) + bn.eps)
        )
        expected_eval = (
            expected_eval * bn.weight.view(1, 2, 1, 1)
            + bn.bias.view(1, 2, 1, 1)
        )

    torch.testing.assert_close(eval_output_1, eval_output_2)
    torch.testing.assert_close(eval_output_1, expected_eval)
    assert torch.equal(bn.running_mean, mean_after)
    assert torch.equal(bn.running_var, var_after)
    assert torch.equal(bn.num_batches_tracked, batches_after)

    # 4. weight/bias 是可训练参数，运行统计是无需梯度的 buffer。
    parameters = dict(bn.named_parameters())
    buffers = dict(bn.named_buffers())
    assert set(parameters) == {"weight", "bias"}
    assert set(buffers) == {"running_mean", "running_var", "num_batches_tracked"}
    assert all(parameter.requires_grad for parameter in parameters.values())
    assert all(not buffer.requires_grad for buffer in buffers.values())

    print("parameters:", list(parameters))
    print("buffers:", list(buffers))
    print("running_mean:", mean_before.tolist(), "->", mean_after.tolist())
    print("running_var:", var_before.tolist(), "->", var_after.tolist())
    print("num_batches_tracked:", batches_before.item(), "->", batches_after.item())

def test_basicblock():
    # 输出通道、步幅、是否投影、预期的单样本输出形状
    cases = [
        (64, 1, False, (64, 32, 32)),
        (128, 1, True, (128, 32, 32)),
        (64, 2, True, (64, 16, 16)),
        (128, 2, True, (128, 16, 16)),
    ]

    for batch_size in (1, 2):
        for out_channels, stride, use_1x1conv, expected_shape in cases:
            # 每组重新创建模型，避免之前的梯度和 BN 统计干扰。
            model = BasicBlock(
                in_channels=64,
                out_channels=out_channels,
                stride=stride,
                use_1x1conv=use_1x1conv,
            ).cpu()

            images = torch.randn(batch_size, 64, 32, 32)
            outputs = model(images)

            assert outputs.shape == (batch_size, *expected_shape)
            assert torch.isfinite(outputs).all().item()

            # 构造标量目标，检查梯度能否通过整个块。
            loss = outputs.square().mean()
            assert torch.isfinite(loss).item()
            loss.backward()

            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, f"{name} 没有梯度"
                assert torch.isfinite(parameter.grad).all().item(), (
                    f"{name} 梯度不有限"
                )

def test_basicblock_identity_shortcut():
    model = BasicBlock(
        in_channels=64,
        out_channels=64,
        stride=1,
        use_1x1conv=False,
    ).cpu().eval()

    images = torch.ones(2, 64, 32, 32)

    with torch.no_grad():
        # 将主分支的卷积输出置零。
        for conv in (model.conv1, model.conv2):
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()

        model.bn1.bias.zero_()
        model.bn2.bias.zero_()

        outputs = model(images)

    torch.testing.assert_close(outputs, images)

def test_resnet_forward_backward():
    for batch_size in (1, 2):
        # 每次新建整网，分别检查单张图片和一个小批次。
        model = ResNet18().cpu().train()
        images = torch.randn(batch_size, 3, 96, 96)
        labels = torch.randint(0, 10, (batch_size,), dtype=torch.long)

        # 前向：每张图片得到 10 个类别的 logits。
        model.zero_grad()
        logits = model(images)
        assert logits.shape == (batch_size, 10)
        assert torch.isfinite(logits).all().item()

        # 损失：使用分类标签计算有限的交叉熵标量。
        loss = torch.nn.functional.cross_entropy(logits, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        # 反向：检查入口、各残差块、投影旁路和分类头的参数。
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(parameter.grad).all().item(), f"{name} 梯度不有限"

        print(f"batch={batch_size}, logits={tuple(logits.shape)}, loss={loss.item():.6f}")

def test_resnet_stage_shapes():
    model = ResNet18().cpu().eval()
    images = torch.randn(2, 3, 96, 96)

    # 键是 model.net 中的模块索引，与计划中的 b1–b5 对应。
    expected_shapes = {
        0: (2, 64, 24, 24),   # b1：入口卷积、BN、ReLU、最大池化
        1: (2, 64, 24, 24),   # b2：stage1，保持通道和尺寸
        2: (2, 128, 12, 12),  # b3：stage2，首块下采样
        3: (2, 256, 6, 6),    # b4：stage3，首块下采样
        4: (2, 512, 3, 3),    # b5：stage4，首块下采样
        5: (2, 512, 1, 1),    # 全局平均池化
        6: (2, 512),          # Flatten
        7: (2, 10),           # 分类层
    }
    assert len(model.net) == len(expected_shapes), "整网模块数量与预期不一致"

    with torch.no_grad():
        for index, layer in enumerate(model.net):
            images = layer(images)
            shape = tuple(images.shape)

            print(f"{index}: {type(layer).__name__} -> {shape}")
            assert shape == expected_shapes[index], (
                f"第 {index} 层形状错误："
                f"预期 {expected_shapes[index]}，实际 {shape}"
            )
