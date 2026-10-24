import numpy as np
class LinearModel:
    def  __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, X):
        return X @ self.w + self.b

if __name__ == "__main__":
    model = LinearModel(np.array([[2.0], [-1.0]]), 0.5)
    y_hat = model.predict(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    print("预测结果:")
    print(y_hat)