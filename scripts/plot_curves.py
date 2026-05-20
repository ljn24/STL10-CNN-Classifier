"""Generate training curve figures (Loss & Accuracy) from history.json.

Usage:
    python -m scripts.plot_curves --config configs/baseline.yaml
    python -m scripts.plot_curves --config configs/advanced.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils import load_config


def load_history(output_dir: Path) -> dict:
    history_path = output_dir / "history.json"
    with open(history_path, encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(history: dict, model_name: str) -> plt.Figure:
    """Create a 2x1 subplot figure with Loss (top) and Accuracy (bottom)."""
    epochs_data = history["epochs"]
    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_acc = [e["train_acc"] for e in epochs_data]
    val_acc = [e["val_acc"] for e in epochs_data]

    best_epoch = history["best_epoch"]
    best_val_acc = history["best_val_acc"]

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Loss subplot ---
    ax_loss.plot(epochs, train_loss, label="Train Loss", linewidth=1.5)
    ax_loss.plot(epochs, val_loss, "--", label="Val Loss", linewidth=1.5)
    best_val_loss = epochs_data[best_epoch - 1]["val_loss"]
    ax_loss.plot(best_epoch, best_val_loss, "*", markersize=14, color="red",
                 label=f"Best epoch {best_epoch}")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.set_title(f"{model_name} — Training Curves")
    ax_loss.grid(True, alpha=0.3)

    # --- Accuracy subplot ---
    ax_acc.plot(epochs, train_acc, label="Train Acc", linewidth=1.5)
    ax_acc.plot(epochs, val_acc, "--", label="Val Acc", linewidth=1.5)
    ax_acc.plot(best_epoch, best_val_acc, "*", markersize=14, color="red",
                label=f"Best epoch {best_epoch} ({best_val_acc:.4f})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from history.json")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    model_name = cfg.get("model_type", "model").capitalize()

    history = load_history(output_dir)
    fig = plot_training_curves(history, model_name)

    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
