"""提供残差块与可切换旁路的 ResNet18 分类器。"""

from torch import nn


class BasicBlock(nn.Module):
    """两层 3×3 卷积组成的残差块，可选 1×1 投影旁路。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_1x1conv=True,
        use_residual=True,
    ):
        """创建主分支和可选的投影卷积。

        Args:
            in_channels: 输入通道数。
            out_channels: 主分支输出通道数。
            stride: 第一层卷积与投影卷积的步幅。
            use_1x1conv: 是否创建 1×1 投影。
            use_residual: 是否在激活前将旁路与主分支相加。
        """
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            if use_1x1conv
            else None
        )

    def forward(self, X):
        """计算主分支并按配置加入旁路，再应用 ReLU。

        Args:
            X: 形状 [B, in_channels, H, W] 的特征图。

        Returns:
            通道数为 out_channels 的特征图，高宽由 stride 决定。
        """
        y = self.relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))
        if self.use_residual:
            if self.shortcut:
                X = self.shortcut(X)
            y += X
        return self.relu(y)


class ResNet18(nn.Module):
    """由四组各含两个残差块的阶段组成的 RGB 图像分类器。"""

    def __init__(self, num_classes=10, use_residual=True):
        """按阶段组装卷积入口、残差块和分类头。

        Args:
            num_classes: 分类头的输出数。
            use_residual: 所有 BasicBlock 是否执行残差相加。
        """

        def resnet_block(
            in_channels,
            out_channels,
            num_residuals,
            first_block=False,
            use_residual=True,
        ):
            """为一个阶段创建有序的残差块列表。

            Args:
                in_channels: 阶段输入通道数。
                out_channels: 阶段输出通道数。
                num_residuals: 块数。
                first_block: 是否为第一阶段。第一阶段保持输入的空间尺寸。
                use_residual: 阶段内各块是否执行残差相加。

            Returns:
                按执行顺序排列的 BasicBlock 列表。
            """
            block = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    block.append(
                        BasicBlock(
                            in_channels,
                            out_channels,
                            stride=2,
                            use_1x1conv=True,
                            use_residual=use_residual,
                        )
                    )
                else:
                    block.append(
                        BasicBlock(
                            out_channels,
                            out_channels,
                            stride=1,
                            use_1x1conv=False,
                            use_residual=use_residual,
                        )
                    )
            return block

        super().__init__()

        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # * 将列表中的块展开为单独参数，Sequential 按列表顺序注册并执行它们。
        b2 = nn.Sequential(
            *resnet_block(
                64, 64, 2, first_block=True, use_residual=use_residual
            )
        )
        b3 = nn.Sequential(*resnet_block(64, 128, 2, use_residual=use_residual))
        b4 = nn.Sequential(
            *resnet_block(128, 256, 2, use_residual=use_residual)
        )
        b5 = nn.Sequential(
            *resnet_block(256, 512, 2, use_residual=use_residual)
        )
        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, X):
        logits = self.net(X)
        return logits
