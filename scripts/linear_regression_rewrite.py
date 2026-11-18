"""独立重写 NumPy 线性回归以复习梯度更新。"""

from pathlib import Path

import matplotlib.pyplot as plt
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


def generate_data(true_w, true_b, seed=42, n_samples=200):
    """根据调用者传入的参数生成带噪声的双特征回归数据。

    Args:
        true_w: 生成标签用的权重列向量，形状 (2, 1)。
        true_b: 生成标签用的标量偏置。
        seed: 独立 NumPy 随机生成器的种子。
        n_samples: 样本数。

    Returns:
        (X, y)，形状分别为 (N, 2) 和 (N, 1)。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 2))
    noise = 0.1 * rng.normal(size=(n_samples, 1))
    y = X @ true_w + true_b + noise
    return X, y


def train(X, y, learning_rate=0.05, steps=500):
    """从零初始化并用整批数据执行梯度下降。

    Args:
        X: 特征矩阵，形状 (N, d)。
        y: 目标列向量，形状 (N, 1)。
        learning_rate: 每次参数更新的步长。
        steps: 参数更新次数。

    Returns:
        (w, b, losses)，losses 依次记录初始 MSE 和每次更新后的 MSE。
    """
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    losses = [mse(y, predict(X, w, b))]
    print(f"Initial loss: {losses[0]}, w: {w.ravel()}, b: {b}")
    print(f"Training for {steps} steps with learning rate {learning_rate}")
    print(
        "-------------------------------- Training Start "
        "--------------------------------"
    )
    for step in range(steps):
        dw, db = gradients(X, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss = mse(y, predict(X, w, b))
        losses.append(loss)
        if (step + 1) % 50 == 0:
            weights_text = ", ".join(f"{value:10.6f}" for value in w.ravel())
            print(
                f"Step {step + 1:4d} | "
                f"Loss: {loss:12.8f} | "
                f"w: [{weights_text}] | "
                f"b: {b:10.6f}"
            )
    print(
        "--------------------------------- Training End "
        "---------------------------------"
    )
    print(f"Final loss: {losses[-1]}, w: {w.ravel()}, b: {b}")
    return w, b, losses


def main():
    """用独立重写的函数训练线性模型并保存损失曲线。"""
    true_w = np.array([[3.0], [-2.0]])
    true_b = 1.0
    X, y = generate_data(true_w, true_b)
    w, b, losses = train(X, y, 0.05, 500)
    steps = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(steps, losses)
    ax.set(xlabel="iterations", ylabel="loss", title="Loss over iterations")
    ax.grid()
    output_path = Path("experiments/basics/week1/figures/loss_curve2.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"True weights: {true_w.ravel()}, True bias: {true_b}")
    print(f"Learned weights: {w.ravel()}, Learned bias: {b}")


if __name__ == "__main__":
    main()
