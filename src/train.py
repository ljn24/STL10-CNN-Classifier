"""Training script for STL-10 classification.

Usage:
    python -m src.train --config configs/baseline.yaml
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from src.dataset import build_train_val_loaders
from src.model import build_model
from src.utils import evaluate, load_config, save_checkpoint, save_json, set_seed

_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
    x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def _cutmix_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    y1, y2, x1, x2 = _rand_bbox(images.size(2), images.size(3), lam)
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (images.size(2) * images.size(3))
    return images, labels, labels[indices], lam


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cutmix_alpha: float | None = None,
) -> tuple[float, float]:
    """Train for one epoch, optionally applying CutMix augmentation."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if cutmix_alpha is not None:
            images, labels_a, labels_b, lam = _cutmix_batch(images, labels, cutmix_alpha)
            logits = model(images)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            correct += (lam * (logits.argmax(1) == labels_a).sum().item()
                        + (1 - lam) * (logits.argmax(1) == labels_b).sum().item())
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STL10 CNN")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    data_cfg = cfg["data"]
    train_loader, val_loader = build_train_val_loaders(
        root=data_cfg["root"],
        val_ratio=data_cfg["val_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
        augment=data_cfg.get("augment", False),
    )

    train_cfg = cfg["train"]
    model = build_model(cfg).to(device)
    epochs = train_cfg["epochs"]

    optimizer = _OPTIMIZERS[train_cfg.get("optimizer", "adam")](
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    sched_name = train_cfg.get("scheduler", "none")
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        if sched_name == "cosine" else None
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    cutmix_alpha = train_cfg.get("cutmix_alpha")

    history: list[dict] = []
    best_val_acc, best_epoch = 0.0, 0
    ckpt_path = output_dir / "best_model.pt"

    t0 = time.time()
    pbar = trange(1, epochs + 1, desc="Training")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            cutmix_alpha=cutmix_alpha,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            save_checkpoint(model, ckpt_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
        })

        pbar.set_postfix_str(
            f"t_acc={train_acc:.4f} v_acc={val_acc:.4f}"
            f"{' *' if val_acc >= best_val_acc else ''}"
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

    save_json({
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 5),
        "total_time_seconds": round(elapsed, 2),
        "epochs": history,
    }, output_dir / "history.json")


if __name__ == "__main__":
    main()
