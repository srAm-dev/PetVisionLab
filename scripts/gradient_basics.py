"""提供 NumPy 线性回归的预测、损失和解析梯度。"""

import numpy as np


def predict(X, w, b):
    """使用当前参数计算线性预测。

    Args:
        X: 特征矩阵，形状 (N, d)。
        w: 权重列向量，形状 (d, 1)。
        b: 标量偏置。

    Returns:
        预测列向量，形状 (N, 1)。
    """
    return X @ w + b


def mse(y_true, y_pred):
    """计算逐元素误差平方的平均值。

    Args:
        y_true: 真实值数组。
        y_pred: 预测值数组，形状须与 y_true 相同。

    Returns:
        标量均方误差。

    Raises:
        ValueError: 两个数组形状不同。
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"标签与预测形状不一致：{y_true.shape} vs {y_pred.shape}"
        )
    return np.mean((y_true - y_pred) ** 2)


def gradients(X, y, w, b):
    """用均方误差的解析公式计算当前点的参数梯度。

    Args:
        X: 特征矩阵，形状 (N, d)。
        y: 目标列向量，形状 (N, 1)。
        w: 权重列向量，形状 (d, 1)。
        b: 标量偏置。

    Returns:
        (dw, db)，dw 与 w 同形状，db 为标量。

    Raises:
        ValueError: 预测与目标的形状不同。
    """
    y_pred = predict(X, w, b)
    if y.shape != y_pred.shape:
        raise ValueError(f"标签与预测形状不一致：{y.shape} vs {y_pred.shape}")
    E = y_pred - y
    N = X.shape[0]
    dw = (2 / N) * X.T @ E
    db = (2 / N) * np.sum(E)
    return dw, db


if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([[1.0], [2.0], [3.0]])
    w = np.array([[0], [0]])
    b = 0.0
    learning_rate = 0.01
    loss0 = mse(y, predict(X, w, b))
    print("初始预测：", predict(X, w, b))
    print("初始损失：", loss0)
    dw, db = gradients(X, y, w, b)
    print("dw：", dw)
    print("db：", db)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    loss1 = mse(y, predict(X, w, b))
    print("更新后的预测：", predict(X, w, b))
    print("更新后的损失：", loss1)

    for i in range(100):
        dw, db = gradients(X, y, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        print(
            f"迭代 {i + 1}，损失：{mse(y, predict(X, w, b))}, "
            f"w：{w.ravel()}, b：{b}, 预测：{predict(X, w, b)}"
        )
