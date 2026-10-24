import numpy as np

# 计算均方误差函数
def mse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"标签与预测形状不一致：{y_true.shape} 和 {y_pred.shape}"
        )
    return np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    from linear_model import LinearModel
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w = np.array([[2.0], [-1.0]])
    b = 0.5
    y_hat = X @ w + b

    print(X.shape, w.shape, y_hat.shape)
    print(y_hat)

    # 预测结果为[0.5, 2.5, 4.5]
    # X有三个样本，每个样本两个特征
    # X的形状为(3, 2)，w的形状为(2, 1)，因此可以相乘，输出形状为(m,n) @ (n,p) → (m,p)。
    # 标量加向量或矩阵时，会被广播为相同形状，因此可以相加

    X1 = X[:1, :] # 取第一行
    print("X第一行:")
    print(X1.shape)
    print(X1)
    print("")
    X2 = X[:, :1] # 取第一列
    print("X第一列:")
    print(X2.shape)
    print(X2)
    print("")
    X3 = X[0:2, :] # 取前两行
    print("X前两行:")
    print(X3.shape)
    print(X3)
    print("")

    # 比较X[:, 0] 与 X[:, 0:1] 的维度
    print(X[:, 0].shape)
    print(X[:, 0:1].shape)
    print("")

    # 比较 X.sum(axis=0) 与 X.sum(axis=1)
    print("X.sum(axis=0):")
    print(X.sum(axis=0).shape)
    print(X.sum(axis=0))
    print("")
    print("X.sum(axis=1):")
    print(X.sum(axis=1).shape)
    print(X.sum(axis=1))
    print("")
    # axis=0 沿样本维求和，得到每列总和；axis=1 沿特征维求和，得到每行总和。

    # 比较 X @ w 与 X * w.T
    print("X @ w:")
    print((X @ w).shape)
    print(X @ w)
    print("")
    print("X * w.T:")
    print((X * w.T).shape)  
    print(X * w.T)
    print("")

    # 把 w 从 (2,1) 变成 (2,)，观察预测输出怎样变化
    w1 = w.flatten()  # 将 w 从 (2,1) 变成 (2,)
    print("w after flattening:")
    print(w1.shape)
    print(w1)
    print("")
    y_hat = X @ w1 + b
    print("y_hat after changing w:")
    print(y_hat.shape)
    print(y_hat)


    #使用MSE函数计算均方误差
    print("均方误差为:", mse(np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 3.0])))
    print("")
    y_true = np.array([1.0, 2.0, 3.0])  # (3,)
    y_pred = X @ w + b                  # (3,1)

    # 观察错误广播：每个预测都与三个标签相减
    print("标签形状：", y_true.shape)
    print("预测形状：", y_pred.shape)
    print("错误相减结果：")
    print(y_true - y_pred)
    print("错误相减形状：", (y_true - y_pred).shape)  # (3,3)

    # 验证 mse 能拦截这种输入
    try:
        mse(y_true, y_pred)
    except ValueError as error:
        print("成功拦截：", error)

    # 修复：把标签变成列向量
    y_true = y_true.reshape(-1, 1)

    print("修复后的标签形状：", y_true.shape)           # (3,1)
    print("修复后的相减形状：", (y_true - y_pred).shape)  # (3,1)
    print("正确 MSE：", mse(y_true, y_pred)) 
    print("")

    # 使用LinearModel类进行预测
    model = LinearModel(w, b)
    y_hat_model = model.predict(X)
    print("使用LinearModel类进行预测:") 
    print(y_hat_model)
