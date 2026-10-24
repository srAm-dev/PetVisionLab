import numpy as np


if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[2.0], [-1.0]])
    b = 0.5

    print("X 的形状：", X.shape)
    print("w 的形状：", w.shape)

    # 每行与 w 做点积，得到该样本的特征加权和。
    # X[0:1] 保留行维度：(1,2) @ (2,1) → (1,1)。
    first_dot = X[0:1] @ w
    print("第一行与 w 的点积：", first_dot)
    print("点积结果的形状：", first_dot.shape)

    # 同时计算三个样本：(3,2) @ (2,1) → (3,1)。
    weighted_sums = X @ w
    y_hat = weighted_sums + b
    print("X @ w：")
    print(weighted_sums)
    print("加上偏置后的预测：")
    print(y_hat)
    print("预测的形状：", y_hat.shape)

    # 转置交换行和列：(3,2) → (2,3)。
    X_transpose = X.T
    print("X.T：")
    print(X_transpose)
    print("转置后的形状：", X_transpose.shape)

    # (2,3) @ (3,2) → (2,2)。
    feature_products = X.T @ X
    print("X.T @ X：")
    print(feature_products)
    print("结果的形状：", feature_products.shape)

    # 用全为 1 的示例数组验证五个特征时的形状。
    X_five = np.ones((3, 5))
    w_five = np.ones((5, 1))
    y_hat_five = X_five @ w_five + b
    print("五个特征：X、w、预测的形状：")
    print(X_five.shape, w_five.shape, y_hat_five.shape)

    # 与手算值比较；数值或形状不一致时会报错。
    np.testing.assert_allclose(first_dot, [[0.0]])
    np.testing.assert_allclose(weighted_sums, [[0.0], [2.0], [4.0]])
    np.testing.assert_allclose(y_hat, [[0.5], [2.5], [4.5]])
    np.testing.assert_allclose(X_transpose, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(feature_products, [[35.0, 44.0], [44.0, 56.0]])
    assert y_hat.shape == (3, 1)
    assert y_hat_five.shape == (3, 1)
    print("矩阵计算与手算结果一致，形状检查通过。")


    def f(w):
        return w**2
    def df(w):
        return 2*w
    learning_rate = 0.1
    w = 2.0

    for i in range(10):
        w = w - learning_rate * df(w)
        print(f"迭代 {i+1}: w = {w}, f(w) = {f(w)}")