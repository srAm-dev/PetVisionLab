import torch
from torch.utils.data import DataLoader, TensorDataset

from petvision.models.cv.mlp import MLP


def train_one_epoch(model, loader, optimizer, criterion, device):
    """练习版：保留原函数名，但这里只训练一个批次。"""
    model.train()
    X, y = next(iter(loader))
    X = X.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    assert pred.shape == (X.shape[0], 10)
    assert torch.isfinite(loss).item(), "训练 loss 出现 NaN 或 Inf"
    loss.backward()
    optimizer.step()
    print(f"train: logits={tuple(pred.shape)}, loss={loss.item():.6f}")


def evaluate(model, loader, criterion, device):
    """练习版：只验证一个批次，不返回汇总指标。"""
    model.eval()
    with torch.no_grad():
        X, y = next(iter(loader))
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        assert pred.shape == (X.shape[0], 10)
        assert torch.isfinite(loss).item(), "验证 loss 出现 NaN 或 Inf"
        assert not pred.requires_grad, "验证输出不应记录梯度"
        print(f"val: logits={tuple(pred.shape)}, loss={loss.item():.6f}")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 合成两批符合 MNIST 形状的数据，只检查代码，不衡量识别效果。
    batch_size = 8
    train_set = TensorDataset(
        torch.randn(batch_size, 1, 28, 28),
        torch.randint(0, 10, (batch_size,)),
    )
    val_set = TensorDataset(
        torch.randn(batch_size, 1, 28, 28),
        torch.randint(0, 10, (batch_size,)),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # 新建临时模型，不读取或保存正式 checkpoint。
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # clone 保存独立副本，避免更新参数时“更新前”的值也随之改变。
    before_train = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    train_one_epoch(model, train_loader, optimizer, criterion, device)
    assert model.training
    changed = [
        name for name, parameter in model.named_parameters()
        if not torch.equal(before_train[name], parameter.detach())
    ]
    assert changed, "训练后没有任何参数发生变化"
    print(f"训练检查通过：{len(changed)} 个参数张量发生变化。")

    before_val = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    evaluate(model, val_loader, criterion, device)
    assert not model.training
    assert all(
        torch.equal(before_val[name], parameter.detach())
        for name, parameter in model.named_parameters()
    ), "验证时不应更新参数"
    print("验证检查通过：全部参数保持不变，输出不记录梯度。")

    # 正式 MNIST 实验：55k 训练集更新参数；5k 验证集选择方案与 best；
    # 10k 测试集在模型和配置确定后评估最终表现。


if __name__ == "__main__":
    main()
