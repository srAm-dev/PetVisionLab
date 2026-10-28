import pytest
import torch
from PIL import Image
from torchvision import transforms

from petvision.data import datasets


class FakeMNIST:
    """只模拟 MNIST 的读图接口，避免测试下载或改动真实数据。"""

    def __init__(self, root, train, download, transform):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 60000 if self.train else 10000

    def __getitem__(self, index):
        image = Image.new("L", (28, 28), color=index % 256)
        return self.transform(image), index % 10


@pytest.fixture
def mnist_args(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "MNIST", FakeMNIST)
    return dict(
        data_dir=tmp_path / "data", split_path=tmp_path / "split.json",
        batch_size=4, seed=42,
    )


def test_mnist_training_transform_does_not_leak_into_evaluation(mnist_args):
    train_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Lambda(lambda image: image + 10),
    ])
    eval_transform = transforms.ToTensor()
    train, val, test = datasets.build_mnist_loaders(
        **mnist_args, train_transform=train_transform, eval_transform=eval_transform,
    )

    assert [len(loader.dataset) for loader in (train, val, test)] == [55000, 5000, 10000]
    assert train.dataset.dataset is not val.dataset.dataset
    train_images, train_labels = next(iter(train))
    assert train_images.shape == (4, 1, 28, 28)
    assert train_images.dtype == torch.float32
    assert train_labels.shape == (4,) and train_labels.dtype == torch.long
    assert train_images.min() >= 10
    val_before = val.dataset[0][0].clone()
    test_before = test.dataset[0][0].clone()
    for loader in (val, test):
        images, _ = next(iter(loader))
        assert images.min() >= 0 and images.max() <= 1

    # 改变训练 Dataset 的 transform 不应影响验证或测试 Dataset。
    train.dataset.dataset.transform = lambda image: torch.full((1, 28, 28), 20.0)
    assert torch.equal(train.dataset[0][0], torch.full((1, 28, 28), 20.0))
    assert torch.equal(val.dataset[0][0], val_before)
    assert torch.equal(test.dataset[0][0], test_before)


def test_mnist_transform_change_reuses_saved_split(mnist_args):
    to_tensor = transforms.ToTensor()
    train, val, _ = datasets.build_mnist_loaders(
        **mnist_args, train_transform=to_tensor, eval_transform=to_tensor,
    )
    split_before = mnist_args["split_path"].read_bytes()
    normalized = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_again, val_again, _ = datasets.build_mnist_loaders(
        **mnist_args, train_transform=normalized, eval_transform=normalized,
    )

    assert train.dataset.indices == train_again.dataset.indices
    assert val.dataset.indices == val_again.dataset.indices
    images, labels = next(iter(train))
    normalized_images, normalized_labels = next(iter(train_again))
    assert torch.equal(labels, normalized_labels)
    torch.testing.assert_close(normalized_images, (images - 0.5) / 0.5, rtol=0, atol=0)
    assert torch.equal(val_again.dataset[0][0], (val.dataset[0][0] - 0.5) / 0.5)

    with pytest.raises(ValueError, match="train_indices 应有"):
        datasets.build_mnist_loaders(
            **mnist_args, train_transform=normalized, eval_transform=normalized,
            validation_size=10000,
        )
    assert mnist_args["split_path"].read_bytes() == split_before