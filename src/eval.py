"""Evaluate a trained model on the STL-10 test set.

Usage:
    python -m src.eval --config configs/baseline.yaml
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import build_test_loader
from src.model import build_model
from src.utils import (
    compute_metrics,
    evaluate,
    load_checkpoint,
    load_config,
    save_json,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STL10 CNN")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    test_loader, class_names = build_test_loader(
        root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    checkpoint = cfg.get("checkpoint", str(output_dir / "best_model.pt"))
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, device=device)

    test_loss, _, y_true, y_pred = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
    )

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["test_loss"] = round(test_loss, 5)

    out_path = output_dir / "test_metrics.json"
    save_json(metrics, out_path)
    print(f"Test accuracy: {metrics['accuracy']:.4f} → {out_path}")


if __name__ == "__main__":
    main()
