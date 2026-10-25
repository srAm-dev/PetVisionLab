import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    """{"loss": 平均损失, "accuracy": 总正确数/总样本数, "n_samples": 实际样本数}"""
    model.train()
    result = {"loss":0.0, "accuracy":0.0, "n_samples":0}
    finalloss, correct = 0, 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        batch_size = y.shape[0]
        finalloss += loss.item() * batch_size
        result["n_samples"] += batch_size
        correct += (pred.argmax(dim=1) == y).sum().item()
        loss.backward()
        optimizer.step()
    if result["n_samples"] == 0:
        raise ValueError("训练数据为空，无法计算指标")
    finalloss /= result["n_samples"]
    correct /= result["n_samples"]
    result["loss"] = finalloss
    result["accuracy"] = correct
    return result

def evaluate(model, loader, criterion, device):
    """{"loss": 平均损失, "accuracy": 总正确数/总样本数, "n_samples": 实际样本数}"""
    model.eval()
    result = {"loss":0.0, "accuracy":0.0, "n_samples":0}
    finalloss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            batch_size = y.shape[0]
            finalloss += loss.item() * batch_size
            result["n_samples"] += batch_size
            correct += (pred.argmax(dim=1) == y).sum().item()
    if result["n_samples"] == 0:
        raise ValueError("验证数据为空，无法计算指标")
    finalloss /= result["n_samples"]
    correct /= result["n_samples"]
    result["loss"] = finalloss
    result["accuracy"] = correct
    return result
