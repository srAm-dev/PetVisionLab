from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, X):
        y = self.relu(self.bn1(self.conv1(X)))
        y = self.bn2(self.conv2(y))
        if self.shortcut:
            X = self.shortcut(X)
        y += X
        return self.relu(y)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        
        def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
            block = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    block.append(BasicBlock(in_channels, out_channels, stride=2, use_1x1conv=True))
                else:
                    block.append(BasicBlock(out_channels, out_channels, stride=1, use_1x1conv=False))
            return block
        super().__init__()

        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, X):
        logits = self.net(X)
        return logits

    