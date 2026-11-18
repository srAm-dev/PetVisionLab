"""提供可配置隐藏层宽度的 MNIST 分类基线。"""

from torch import nn


class MLP(nn.Module):
    """将图片展平后经过 ReLU 隐藏层和线性分类头，返回每类的原始分数。"""

    def __init__(self, input_dim=784, hidden_dims=(256, 128), num_classes=10):
        """创建全连接分类网络。

        Args:
            input_dim: 每张图片展平后的特征数。
            hidden_dims: 从输入到输出排列的隐藏层宽度，至少包含一层。
            num_classes: 分类头的输出数。

        Raises:
            ValueError: hidden_dims 为空。
        """
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("MLP 需要隐藏层尺寸")
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()
        )
        # 用当前层数作名称，追加层后 state_dict 仍按 Sequential 的数字索引命名。
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
