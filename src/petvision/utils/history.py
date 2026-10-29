import csv
from pathlib import Path
import matplotlib.pyplot as plt

HISTORY_COLUMNS = [
    "epoch", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy",
]

def save_history(history, experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(history)

def append_history(row, experiment_dir):
    path = Path(experiment_dir) / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def plot_history(history, experiment_dir, variant=None):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set(title="Loss", ylabel="Cross entropy")
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val")
    axes[1].set(title="Accuracy", ylabel="Accuracy", ylim=(0, 1))
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)
    if variant is not None:
        fig.suptitle(variant)
    fig.tight_layout()

    figure_dir = Path(experiment_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "training_curves.png", dpi=150)
    plt.close(fig)
