"""Training script for STL-10 classification.

Usage:
    python -m src.train --config configs/baseline.yaml
"""

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import build_train_val_loaders
from src.model import STL10LiteVGG
from src.utils import (
    evaluate,
    load_config,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STL10-LiteVGG")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    # ── Data ──
    data_cfg = cfg["data"]
    train_loader, val_loader = build_train_val_loaders(
        root=data_cfg["root"],
        val_ratio=data_cfg["val_ratio"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # ── Model ──
    model = STL10LiteVGG(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # ── Training loop ──
    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    ckpt_path = output_dir / "best_model.pt"

    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, ckpt_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
        })

        mark = " *" if improved else ""
        tqdm.write(
            f"Epoch {epoch:>3d}/{cfg['train']['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{mark}"
        )

    total_time = time.time() - t0
    print(f"\nTraining finished in {total_time:.1f}s. "
          f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}.")

    save_json({
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 5),
        "total_time_seconds": round(total_time, 2),
        "epochs": history,
    }, output_dir / "history.json")

    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
