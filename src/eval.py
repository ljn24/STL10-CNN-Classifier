"""Evaluate a trained model on the STL-10 test set.

Usage:
    python -m src.eval --config configs/eval.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_test_loader
from src.model import STL10LiteVGG
from src.utils import (
    compute_metrics,
    evaluate,
    load_checkpoint,
    load_config,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STL10-LiteVGG")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    data_cfg = cfg["data"]
    test_loader, class_names = build_test_loader(
        root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # ── Model ──
    model = STL10LiteVGG(num_classes=cfg["model"]["num_classes"]).to(device)
    load_checkpoint(model, cfg["checkpoint"], device=device)
    print(f"Loaded checkpoint: {cfg['checkpoint']}")

    # ── Evaluate ──
    criterion = nn.CrossEntropyLoss()
    test_loss, _, y_true, y_pred = evaluate(
        model, test_loader, criterion, device,
    )

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["test_loss"] = round(test_loss, 5)

    out_path = output_dir / "test_metrics.json"
    save_json(metrics, out_path)

    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
