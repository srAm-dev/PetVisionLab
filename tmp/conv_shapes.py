import torch
from torch import nn

images = torch.randn(4, 3, 32, 32)
conv_s1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
conv_s2 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

with torch.no_grad():
    output_s1 = conv_s1(images)
    output_s2 = conv_s2(images)  # 分别作用于原始图片，不串联两个卷积。
    pooled = pool(output_s1)   # 池化接在 stride=1 的卷积后面。

print("输入：", images.shape)
print("stride=1 卷积：", output_s1.shape)
print("stride=2 卷积：", output_s2.shape)
print("stride=1 卷积后池化：", pooled.shape)

assert output_s1.shape == (4, 32, 32, 32)
assert output_s2.shape == (4, 32, 16, 16)
assert pooled.shape == (4, 32, 16, 16)