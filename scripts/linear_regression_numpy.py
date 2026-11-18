"""用合成数据训练基准线性模型，并提供可复用的训练函数。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gradient_basics import gradients, mse, predict


def generate_data(seed=42, n_samples=200):
    """用固定线性关系与高斯噪声生成回归数据。

    Args:
        seed: 独立 NumPy 随机生成器的种子。
        n_samples: 样本数。

    Returns:
        (X, y, true_w, true_b)，前三项形状分别为 (N, 2)、(N, 1)、(2, 1)，
        true_b 为标量。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 2))
    true_w = np.array([[3.0], [-2.0]])
    true_b = 1.0
    noise = 0.1 * rng.normal(size=(n_samples, 1))
    y = X @ true_w + true_b + noise
    return X, y, true_w, true_b


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
    """生成数据、训练基准线性模型并保存损失曲线。"""
    X, y, true_w, true_b = generate_data()
    w, b, losses = train(X, y, 0.05, 500)
    steps = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(steps, losses)
    ax.set(xlabel="iterations", ylabel="loss", title="Loss over iterations")
    ax.grid()
    output_path = Path("experiments/basics/week1/figures/loss_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"True weights: {true_w.ravel()}, True bias: {true_b}")
    print(f"Learned weights: {w.ravel()}, Learned bias: {b}")


if __name__ == "__main__":
    main()
