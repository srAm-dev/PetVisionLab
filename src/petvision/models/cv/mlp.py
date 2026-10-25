from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(256, 128), num_classes=10):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("MLP 需要隐藏层尺寸")
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        for index in range(len(hidden_dims) - 1):
            self.net.add_module(
                str(len(self.net)),
                nn.Linear(hidden_dims[index], hidden_dims[index + 1]),
            )
            self.net.add_module(str(len(self.net)), nn.ReLU())
        self.net.add_module(
            str(len(self.net)), nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, images):
        logits = self.net(images)
        return logits

if __name__ == "__main__":
    print(MLP())
