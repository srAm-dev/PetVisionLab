import numpy as np
def predict(X, w, b):
    """返回形状为 (N,1) 的预测。"""
    return X @ w + b

def mse(y_true, y_pred):
    """检查两者 shape 一致，返回平均平方误差。"""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"标签与预测形状不一致：{y_true.shape} vs {y_pred.shape}"
        )
    return np.mean((y_true - y_pred) ** 2)

def gradients(X, y, w, b):
    """返回 dw 和标量 db。"""
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
        print(f"迭代 {i+1}，损失：{mse(y, predict(X, w, b))}, w：{w.ravel()}, b：{b}, 预测：{predict(X, w, b)}")