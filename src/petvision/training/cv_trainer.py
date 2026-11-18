"""提供图像分类的训练与评估循环，按实际样本数汇总指标。"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    """切换到 train 模式，遍历传入批次并逐批更新分类模型。

    Args:
        model: 返回分类 logits 的模型。
        loader: 产生 (images, labels) 的可迭代对象。
        optimizer: 绑定模型参数的优化器。
        criterion: 返回当前批次平均损失的损失函数。
        device: 图片和标签迁移到的设备。

    Returns:
        字典，包含按样本平均的 loss、accuracy 和实际样本数 n_samples。

    Raises:
        ValueError: 没有处理到任何样本。
    """
    model.train()
    result = {"loss": 0.0, "accuracy": 0.0, "n_samples": 0}
    finalloss, correct = 0, 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        batch_size = y.shape[0]
        # 每批 loss 是均值，乘实际批大小后才能汇总最后一个小批次。
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
    """在 eval 模式下关闭梯度，遍历传入批次并统计分类指标。

    调用后模型保持 eval 模式。

    Args:
        model: 返回分类 logits 的模型。
        loader: 产生 (images, labels) 的可迭代对象。
        criterion: 返回当前批次平均损失的损失函数。
        device: 图片和标签迁移到的设备。

    Returns:
        字典，包含按样本平均的 loss、accuracy 和实际样本数 n_samples。

    Raises:
        ValueError: 没有处理到任何样本。
    """
    model.eval()
    result = {"loss": 0.0, "accuracy": 0.0, "n_samples": 0}
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
