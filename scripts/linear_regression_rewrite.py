import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def predict(X, w, b):
    return X @ w + b

def mse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"标签与预测形状不一致：{y_true.shape} vs {y_pred.shape}"
        )
    return np.mean((y_true - y_pred) ** 2)

def gradients(X, y, w, b):
    y_pred = predict(X, w, b)
    if y.shape != y_pred.shape:
        raise ValueError(
            f"标签与预测形状不一致：{y.shape} vs {y_pred.shape}"
        )
    E = y_pred - y
    N = X.shape[0]
    dw = (2 / N) * X.T @ E
    db = (2 / N) * np.sum(E)
    return dw, db

def generate_data(true_w, true_b, seed=42, n_samples=200):
    """生成线性数据集。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples,2))
    noise = 0.1 * rng.normal(size=(n_samples, 1))
    y = X @ true_w + true_b +noise
    return X, y

def train(X, y, learning_rate=0.05, steps=500):
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    losses = [mse(y, predict(X, w, b))]
    print(f"Initial loss: {losses[0]}, w: {w.ravel()}, b: {b}")
    print(f"Training for {steps} steps with learning rate {learning_rate}")
    print('-------------------------------- Training Start --------------------------------')
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
    print('--------------------------------- Training End ---------------------------------')
    print(f"Final loss: {losses[-1]}, w: {w.ravel()}, b: {b}")
    return w, b, losses

def main():
    true_w = np.array([[3.0], [-2.0]])
    true_b = 1.0
    X, y = generate_data(true_w, true_b)
    w, b, losses = train(X, y, 0.05, 500)
    steps = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(steps, losses)
    ax.set(xlabel='iterations', ylabel='loss',
        title='Loss over iterations')
    ax.grid()
    output_path = Path("experiments/basics/week1/figures/loss_curve2.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"True weights: {true_w.ravel()}, True bias: {true_b}")
    print(f"Learned weights: {w.ravel()}, Learned bias: {b}")


if __name__ == "__main__":
    main()