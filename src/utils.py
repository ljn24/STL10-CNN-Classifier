import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device | None = None) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Return accuracy, per-class precision/recall/F1, and confusion matrix."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": class_names,
    }


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Return (avg_loss, accuracy, all_labels, all_preds)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, all_labels, all_preds
