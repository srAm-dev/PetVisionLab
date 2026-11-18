"""用线性预测演示类的实例属性与方法。"""

import numpy as np


class LinearModel:
    """保存线性模型的参数，并用它们预测一批样本。

    Attributes:
        w: 形状 (d, 1) 的权重列向量。
        b: 用于所有样本的标量偏置。
    """

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, X):
        """计算 X @ w + b。

        Args:
            X: 特征矩阵，形状 (N, d)。

        Returns:
            预测列向量，形状 (N, 1)。
        """
        return X @ self.w + self.b


if __name__ == "__main__":
    model = LinearModel(np.array([[2.0], [-1.0]]), 0.5)
    y_hat = model.predict(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    print("预测结果:")
    print(y_hat)
