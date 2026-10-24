import numpy as np

from linear_regression_numpy import generate_data, gradients, mse, predict


def check_gradients(X, y, w, b, epsilon=1e-6):
    """用中心差分逐个核对权重和偏置，返回各参数的比较结果。"""
    # 解析梯度：使用 Day 4 推导的公式，在原始参数处计算一次。
    dw, db = gradients(X, y, w, b)
    results = []

    for index in range(w.shape[0]):
        # 每次复制原始参数，只改变一个权重，其他权重与 b 保持不变。
        # 使用副本可以保证检查结束后，原始 w 仍然保持原值。
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[index, 0] += epsilon
        w_minus[index, 0] -= epsilon

        loss_plus = mse(y, predict(X, w_plus, b))
        loss_minus = mse(y, predict(X, w_minus, b))
        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
        analytical_gradient = dw[index, 0]

        results.append({
            "parameter": f"w{index + 1}",
            "analytical_gradient": float(analytical_gradient),
            "numerical_gradient": float(numerical_gradient),
            "absolute_error": float(abs(analytical_gradient - numerical_gradient)),
        })

    # 检查偏置：w 保持原值，仅将 b 分别增加和减少 epsilon。
    loss_plus = mse(y, predict(X, w, b + epsilon))
    loss_minus = mse(y, predict(X, w, b - epsilon))
    numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
    results.append({
        "parameter": "b",
        "analytical_gradient": float(db),
        "numerical_gradient": float(numerical_gradient),
        "absolute_error": float(abs(db - numerical_gradient)),
    })
    return results


def main():
    epsilon = 1e-6
    tolerance = 1e-5

    # 固定 5 条样本；使用 float64 和非零参数来检查当前点的梯度。
    X, y, _, _ = generate_data(seed=42, n_samples=5)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    w = np.array([[0.3], [-0.7]], dtype=np.float64)
    b = np.float64(0.2)

    results = check_gradients(X, y, w, b, epsilon)
    print(f"样本数：{X.shape[0]}，数据类型：{X.dtype}")
    print(f"检查位置：w = {w.ravel()}，b = {b}")
    print(f"epsilon = {epsilon:.0e}，误差阈值 = {tolerance:.0e}")
    print("parameter |     analytical |      numerical | absolute error")
    for result in results:
        print(
            f"{result['parameter']:>9} | "
            f"{result['analytical_gradient']:14.8f} | "
            f"{result['numerical_gradient']:14.8f} | "
            f"{result['absolute_error']:14.6e}"
        )

    max_error = float(np.max([result["absolute_error"] for result in results]))
    print(f"最大绝对误差：{max_error:.6e}")

    if not max_error < tolerance:
        raise AssertionError("梯度检查未通过，请检查 MSE、梯度公式和数组形状。")
    print("梯度检查通过：两个权重和偏置的误差均小于阈值。")


if __name__ == "__main__":
    main()
