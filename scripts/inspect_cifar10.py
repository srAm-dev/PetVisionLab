import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from petvision.data.datasets import build_cifar10_loaders

def main():
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader, val_loader, test_loader = build_cifar10_loaders(
            data_dir="datasets/raw",
            split_path="experiments/cnn_vs_resnet/week3/cifar10_split.json",
            batch_size=64,
            seed=42,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    assert len(train_loader.dataset) == 45000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000

    images, labels = next(iter(train_loader))
    print("images:", images.shape, images.dtype)
    print("image range:", images.min().item(), images.max().item())
    print("labels:", labels.shape, labels.dtype)
    print("label range:", labels.min().item(), labels.max().item())
    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all().item()
    assert images.min().item() >= -1 and images.max().item() <= 1
    assert labels.min().item() >= 0 and labels.max().item() <= 9
    assert train_loader.dataset.dataset is not val_loader.dataset.dataset

    image_a, label_a = val_loader.dataset[0]
    image_b, label_b = val_loader.dataset[0]
    assert torch.equal(image_a, image_b)
    assert label_a == label_b
    classes = train_loader.dataset.dataset.classes
    print("classes:", classes)
    
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for index, ax in enumerate(axes.flat):
        image = images[index].cpu() * 0.5 + 0.5
        image = image.permute(1, 2, 0)
        ax.imshow(image.numpy())
        ax.set_title(str(labels[index].item()))
        ax.axis("off")
    fig.tight_layout()
    figure_path = Path("experiments/cnn_vs_resnet/week3/figures/cifar10_batch.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    print("figure saved:", figure_path)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()