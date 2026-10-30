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