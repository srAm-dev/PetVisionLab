import csv
from pathlib import Path

import matplotlib.pyplot as plt

from linear_regression_numpy import generate_data, train


def main():
    # 数据只生成一次，三组实验共用同一份 X 和 y。
    X, y, _, _ = generate_data(seed=42, n_samples=200)
    learning_rates = (0.001, 0.05, 0.5)
    steps = 500
    results = []
    fig, ax = plt.subplots(figsize=(8, 5))

    for learning_rate in learning_rates:
        print(f"\n学习率对照：learning_rate = {learning_rate}")
        # train 每次调用都会重新将 w 和 b 初始化为零。
        w, b, losses = train(X, y, learning_rate=learning_rate, steps=steps)
        results.append({
            "learning_rate": learning_rate,
            "steps": steps,
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "w1": float(w[0, 0]),
            "w2": float(w[1, 0]),
            "b": float(b),
        })

        # 包括第 0 步的初始损失，每组曲线都有 501 个点。
        ax.plot(range(len(losses)), losses, label=f"lr = {learning_rate}")

    output_dir = Path("experiments/basics/week1")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "learning_rate", "steps", "initial_loss", "final_loss", "w1", "w2", "b"
        ])
        writer.writeheader()
        writer.writerows(results)

    ax.set(
        xlabel="Update step",
        ylabel="Training MSE",
        title="Learning rate comparison (500 updates)",
    )
    # 对数纵轴便于同时观察初期下降和接近收敛后的损失。
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    figure_path = figures_dir / "learning_rates.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n学习率对照结果：")
    print("learning rate | initial loss |   final loss |         w1 |         w2 |          b")
    for result in results:
        print(
            f"{result['learning_rate']:13.3f} | "
            f"{result['initial_loss']:12.8f} | "
            f"{result['final_loss']:12.8f} | "
            f"{result['w1']:10.6f} | "
            f"{result['w2']:10.6f} | "
            f"{result['b']:10.6f}"
        )
    print(f"结果表已保存：{csv_path}")
    print(f"对照曲线已保存：{figure_path}")


if __name__ == "__main__":
    main()
