import torch
import numpy as np

def main():  
    X = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float64)
    y = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64)
    w = torch.zeros((2, 1), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    loss = ((X @ w + b - y) ** 2).mean()
    loss.backward()
    print("loss:", loss.item())
    print("autograd dw:", w.grad)
    print("autograd db:", b.grad)

    X_np = X.numpy()
    y_np = y.numpy()
    w_np = w.detach().numpy()
    b_np = b.detach().numpy()
    residual_np = X_np @ w_np + b_np - y_np
    loss_np = np.mean(residual_np ** 2)
    dw_np = (2 / X_np.shape[0]) * X_np.T @ residual_np
    db_np = (2 / X_np.shape[0]) * np.sum(residual_np)
    print("loss_np:", loss_np)
    print("numpy dw:", dw_np)
    print("numpy db:", db_np)
    loss_error = abs(loss.item() - loss_np)
    dw_error = np.max(np.abs(w.grad.detach().cpu().numpy() - dw_np))
    db_error = abs(b.grad.item() - db_np)
    print("loss error:", loss_error)
    print("dw max abs error:", dw_error)
    print("db abs error:", db_error)
    assert loss_error < 1e-8
    assert dw_error < 1e-8
    assert db_error < 1e-8

    w = torch.tensor([[0.3],[-0.7]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    loss = ((X @ w + b - y) ** 2).mean()
    loss.backward()
    print("loss:", loss.item())
    print("autograd dw:", w.grad)
    print("autograd db:", b.grad)

    X_np = X.numpy()
    y_np = y.numpy()
    w_np = w.detach().numpy()
    b_np = b.detach().numpy()
    residual_np = X_np @ w_np + b_np - y_np
    loss_np = np.mean(residual_np ** 2)
    dw_np = (2 / X_np.shape[0]) * X_np.T @ residual_np
    db_np = (2 / X_np.shape[0]) * np.sum(residual_np)
    print("loss_np:", loss_np)
    print("numpy dw:", dw_np)
    print("numpy db:", db_np)

    loss_error = abs(loss.item() - loss_np)
    dw_error = np.max(np.abs(w.grad.detach().cpu().numpy() - dw_np))
    db_error = abs(b.grad.item() - db_np)
    print("loss error:", loss_error)
    print("dw max abs error:", dw_error)
    print("db abs error:", db_error)
    assert loss_error < 1e-8
    assert dw_error < 1e-8
    assert db_error < 1e-8

    first_dw = w.grad.clone()
    first_db = b.grad.clone()

    loss2 = ((X @ w + b - y) ** 2).mean()
    loss2.backward()
    print("second dw:", w.grad)
    print("second db:", b.grad)
    dw_error = (w.grad - 2 * first_dw).abs().max().item()
    db_error = abs(b.grad - 2*first_db)
    print("loss2:", loss2.item())
    print("dw max abs error:", dw_error)
    print("db abs error:", db_error)
    assert dw_error < 1e-8
    assert db_error < 1e-8

    w.grad = None
    b.grad = None
    print("after reset:", w.grad, b.grad)
    loss3 = ((X @ w + b - y) ** 2).mean()
    loss3.backward()
    print("third dw:", w.grad)
    print("third db:", b.grad)
    dw_error = (w.grad - first_dw).abs().max().item()
    db_error = abs(b.grad - first_db)
    print("loss3:", loss3.item())
    print("dw max abs error:", dw_error)
    print("db abs error:", db_error)
    assert dw_error < 1e-8
    assert db_error < 1e-8


    print("X.shape:", X.shape)
    print("X.dtype:", X.dtype)
    print("X.device:", X.device)
    print("X.requires_grad:", X.requires_grad)
    print("X.is_leaf:", X.is_leaf)

    print("y.shape:", y.shape)
    print("y.dtype:", y.dtype)
    print("y.device:", y.device)
    print("y.requires_grad:", y.requires_grad)
    print("y.is_leaf:", y.is_leaf)

    print("w.shape:", w.shape)
    print("w.dtype:", w.dtype)
    print("w.device:", w.device)
    print("w.requires_grad:", w.requires_grad)
    print("w.is_leaf:", w.is_leaf)

    print("b.shape:", b.shape)
    print("b.dtype:", b.dtype)
    print("b.device:", b.device)
    print("b.requires_grad:", b.requires_grad)
    print("b.is_leaf:", b.is_leaf)

    prediction = X @ w + b
    detached_prediction = prediction.detach()
    print("prediction requires_grad:", prediction.requires_grad)
    print("detached requires_grad:", detached_prediction.requires_grad)
    print("original still requires_grad:", prediction.requires_grad)
    prediction_np = detached_prediction.cpu().numpy()
    print("NumPy prediction:", prediction_np, prediction_np.shape)

    assert w.grad is not None and b.grad is not None
    assert torch.isfinite(w.grad).all().item()
    assert torch.isfinite(b.grad).all().item()

if __name__ == "__main__":
    main()