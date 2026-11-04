"""Week3 Day6：运行重写的残差块，检查形状、反向传播及原实现对照。"""

import torch
import torch.nn as nn
from petvision.models.cv.resnet import BasicBlock as ReferenceBasicBlock


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=True, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, X):
        y = self.relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))
        if self.use_residual:
            if self.shortcut is not None:
                X = self.shortcut(X)
            y += X
        return self.relu(y)


def check_block(name, out_channels, stride, use_1x1conv, expected_shape, device):
    # 被检查的是本文件中的重写类；两个例子都保留残差相加。
    block = BasicBlock(
        64, out_channels, stride=stride,
        use_1x1conv=use_1x1conv, use_residual=True,
    ).to(device).train()
    reference = ReferenceBasicBlock(
        64, out_channels, stride=stride,
        use_1x1conv=use_1x1conv, use_residual=True,
    ).to(device).train()
    reference.load_state_dict(block.state_dict(), strict=True)

    X = torch.randn(2, 64, 32, 32, device=device, requires_grad=True)
    reference_X = X.detach().clone().requires_grad_(True)

    # hook 只记录实际 forward 的出口形状，不额外执行卷积或更新 BN。
    shapes = {}

    def remember_main_shape(module, inputs, output):
        shapes["main"] = tuple(output.shape)

    def remember_shortcut_shape(module, inputs, output):
        shapes["shortcut"] = tuple(output.shape)

    handles = [block.bn2.register_forward_hook(remember_main_shape)]
    if block.shortcut is None:
        shapes["shortcut"] = tuple(X.shape)
    else:
        handles.append(block.shortcut.register_forward_hook(remember_shortcut_shape))
    try:
        output = block(X)
    finally:
        for handle in handles:
            handle.remove()

    assert shapes["main"] == shapes["shortcut"] == tuple(output.shape) == expected_shape
    assert torch.isfinite(output).all()
    reference_output = reference(reference_X)
    torch.testing.assert_close(output, reference_output)

    # 对真实输出构造标量目标，检查输入和所有活跃参数的梯度。
    loss = output.square().mean()
    reference_loss = reference_output.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    reference_loss.backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert reference_X.grad is not None and torch.isfinite(reference_X.grad).all()
    torch.testing.assert_close(X.grad, reference_X.grad)

    reference_parameters = dict(reference.named_parameters())
    max_parameter_grad_diff = 0.0
    checked_parameters = []
    for parameter_name, parameter in block.named_parameters():
        reference_parameter = reference_parameters[parameter_name]
        assert parameter.grad is not None, f"{parameter_name} 缺少梯度"
        assert torch.isfinite(parameter.grad).all(), f"{parameter_name} 梯度非有限"
        assert reference_parameter.grad is not None, parameter_name
        assert torch.isfinite(reference_parameter.grad).all(), parameter_name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        max_parameter_grad_diff = max(
            max_parameter_grad_diff,
            (parameter.grad - reference_parameter.grad).abs().max().item(),
        )
        checked_parameters.append(parameter_name)

    print(f"\n{name}")
    print(f"  stride={stride}, use_1x1conv={use_1x1conv}, use_residual=True")
    print(f"  输入: {tuple(X.shape)}")
    print(f"  主分支: {shapes['main']}, 旁路: {shapes['shortcut']}")
    print(f"  输出: {tuple(output.shape)}, loss={loss.item():.8f}")
    print(f"  活跃参数梯度: {len(checked_parameters)} 项均存在且有限")
    print(f"  已检查参数: {', '.join(checked_parameters)}")
    print(f"  输入梯度: 存在且有限，norm={X.grad.norm().item():.8f}")
    print(f"  与原实现最大输出差: {(output - reference_output).abs().max().item():.8e}")
    print(f"  与原实现最大输入梯度差: {(X.grad - reference_X.grad).abs().max().item():.8e}")
    print(f"  与原实现最大参数梯度差: {max_parameter_grad_diff:.8e}")
    print("  检查通过")


def main():
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, seed: {seed}, dtype: torch.float32")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    check_block(
        "例 A：普通块（恒等旁路）",
        out_channels=64, stride=1, use_1x1conv=False,
        expected_shape=(2, 64, 32, 32), device=device,
    )
    check_block(
        "例 B：下采样块（1×1 投影旁路）",
        out_channels=128, stride=2, use_1x1conv=True,
        expected_shape=(2, 128, 16, 16), device=device,
    )
    print("\n两种重写块的形状、梯度及原实现对照检查全部通过。")


if __name__ == "__main__":
    main()
