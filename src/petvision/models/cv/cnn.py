from torch import nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, use_bn=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), 
            nn.BatchNorm2d(32) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self, images):
        logits = self.net(images)
        return logits

if __name__ == "__main__":
    print(CNN())
