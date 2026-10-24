from pathlib import Path

import matplotlib.pyplot as plt
import torch

from petvision.data.datasets import build_mnist_loaders


def main():
    train_loader, val_loader, test_loader = build_mnist_loaders(
        data_dir="datasets/raw",
        split_path="experiments/mnist_mlp/week2/mnist_split.json",
        batch_size=128,
        seed=42,
    )
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    assert len(train_loader.dataset) == 55000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000

    images, labels = next(iter(train_loader))
    print("images:", images.shape, images.dtype)
    print("image range:", images.min().item(), images.max().item())
    print("labels:", labels.shape, labels.dtype)
    print("label range:", labels.min().item(), labels.max().item())
    assert images.shape == (128, 1, 28, 28)
    assert labels.shape == (128,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all().item()
    assert images.min().item() >= -1 and images.max().item() <= 1
    assert labels.min().item() >= 0 and labels.max().item() <= 9

    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for index, ax in enumerate(axes.flat):
        image = images[index].cpu() * 0.5 + 0.5
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(labels[index].item()))
        ax.axis("off")
    fig.tight_layout()
    figure_path = Path("experiments/mnist_mlp/week2/figures/mnist_batch.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    print("figure saved:", figure_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
