"""Generate Grad-CAM report figures for trained STL-10 models.

Usage:
    # Single-model figures (class overview + correct vs wrong):
    python -m visualization.visualize --config configs/advanced.yaml

    # Two-model comparison (adds model_comparison.png):
    python -m visualization.visualize \
        --config configs/advanced.yaml \
        --compare configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, _BASE_TRANSFORM
from src.model import build_model
from src.utils import load_checkpoint, load_config, set_seed

from visualization.gradcam import GradCAM
from visualization.plotting import (
    denormalize,
    plot_class_overview,
    plot_correct_vs_wrong,
    plot_model_comparison,
)


def _load_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    output_dir = Path(cfg["output_dir"])
    checkpoint = cfg.get("checkpoint", str(output_dir / "best_model.pt"))
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, device=device)
    model.eval()
    return model


def _pick_samples_per_class(
    dataset: ImageFolder,
    n_per_class: int,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Return ``{class_idx: [dataset_indices]}`` with *n_per_class* per class."""
    rng = random.Random(seed)
    class_to_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.imgs):
        class_to_indices.setdefault(label, []).append(idx)
    for indices in class_to_indices.values():
        rng.shuffle(indices)
    return {
        cls: indices[:n_per_class]
        for cls, indices in class_to_indices.items()
    }


def _run_gradcam_on_indices(
    gradcam: GradCAM,
    dataset: ImageFolder,
    indices: list[int],
    device: torch.device,
) -> list[dict]:
    """Run Grad-CAM on a list of dataset indices and return result dicts."""
    results = []
    for idx in indices:
        img_tensor, true_label = dataset[idx]
        input_t = img_tensor.unsqueeze(0).to(device)
        cam, pred_class, confidence = gradcam.generate(input_t)

        results.append({
            "index": idx,
            "original": denormalize(img_tensor),
            "cam": cam.cpu().numpy(),
            "true_label": true_label,
            "pred_label": pred_class,
            "confidence": confidence,
            "class_name": dataset.classes[true_label],
        })
    return results


def generate_class_overview(
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Figure 1: one correctly-classified sample per class."""
    model = _load_model(cfg, device)
    gradcam = GradCAM(model, model.features)
    dataset = ImageFolder(Path(cfg["data"]["root"]) / "test", transform=_BASE_TRANSFORM)

    samples = _pick_samples_per_class(dataset, n_per_class=10, seed=seed)

    rows = []
    for cls_idx in sorted(samples.keys()):
        for idx in samples[cls_idx]:
            img_tensor, true_label = dataset[idx]
            input_t = img_tensor.unsqueeze(0).to(device)
            cam, pred_class, confidence = gradcam.generate(input_t)
            if pred_class == true_label:
                rows.append({
                    "class_name": dataset.classes[cls_idx],
                    "original": denormalize(img_tensor),
                    "cam": cam.cpu().numpy(),
                    "pred_class": pred_class,
                    "confidence": confidence,
                })
                break

    gradcam.remove_hooks()

    fig = plot_class_overview(rows, title=f"Grad-CAM — {cfg['model_type'].title()} Model")
    path = output_dir / "class_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig)
    print(f"  Saved {path}")


def generate_correct_vs_wrong(
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    n_correct: int = 5,
    n_wrong: int = 5,
    seed: int = 42,
) -> None:
    """Figure 3: correct predictions vs misclassifications."""
    model = _load_model(cfg, device)
    gradcam = GradCAM(model, model.features)
    dataset = ImageFolder(Path(cfg["data"]["root"]) / "test", transform=_BASE_TRANSFORM)

    all_indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    correct_rows: list[dict] = []
    wrong_rows: list[dict] = []

    for idx in all_indices:
        if len(correct_rows) >= n_correct and len(wrong_rows) >= n_wrong:
            break

        img_tensor, true_label = dataset[idx]
        input_t = img_tensor.unsqueeze(0).to(device)
        cam, pred_class, confidence = gradcam.generate(input_t)

        row = {
            "original": denormalize(img_tensor),
            "cam": cam.cpu().numpy(),
            "true_label": true_label,
            "pred_label": pred_class,
            "confidence": confidence,
        }
        if pred_class == true_label and len(correct_rows) < n_correct:
            correct_rows.append(row)
        elif pred_class != true_label and len(wrong_rows) < n_wrong:
            wrong_rows.append(row)

    gradcam.remove_hooks()

    fig = plot_correct_vs_wrong(
        correct_rows, wrong_rows, dataset.classes,
        title=f"Correct vs Wrong — {cfg['model_type'].title()} Model",
    )
    path = output_dir / "correct_vs_wrong.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig)
    print(f"  Saved {path}")


def generate_model_comparison(
    cfg_main: dict,
    cfg_compare: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Figure 2: same images, two models side-by-side."""
    model_main = _load_model(cfg_main, device)
    model_comp = _load_model(cfg_compare, device)
    gc_main = GradCAM(model_main, model_main.features)
    gc_comp = GradCAM(model_comp, model_comp.features)

    dataset = ImageFolder(
        Path(cfg_main["data"]["root"]) / "test", transform=_BASE_TRANSFORM,
    )
    samples = _pick_samples_per_class(dataset, n_per_class=1, seed=seed)

    rows = []
    for cls_idx in sorted(samples.keys()):
        idx = samples[cls_idx][0]
        img_tensor, true_label = dataset[idx]
        input_t = img_tensor.unsqueeze(0).to(device)

        cam_main, _, _ = gc_main.generate(input_t)
        cam_comp, _, _ = gc_comp.generate(input_t)

        main_type = cfg_main["model_type"]
        comp_type = cfg_compare["model_type"]

        row = {
            "class_name": dataset.classes[cls_idx],
            "original": denormalize(img_tensor),
        }
        if comp_type == "baseline":
            row["cam_baseline"] = cam_comp.cpu().numpy()
            row["cam_advanced"] = cam_main.cpu().numpy()
        else:
            row["cam_baseline"] = cam_main.cpu().numpy()
            row["cam_advanced"] = cam_comp.cpu().numpy()
        rows.append(row)

    gc_main.remove_hooks()
    gc_comp.remove_hooks()

    fig = plot_model_comparison(rows)
    path = output_dir / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt_close(fig)
    print(f"  Saved {path}")


def plt_close(fig):
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualisations for STL-10 models",
    )
    parser.add_argument("--config", required=True, help="Primary model config YAML")
    parser.add_argument(
        "--compare", default=None,
        help="Optional second config for model comparison figure",
    )
    parser.add_argument("--n-correct", type=int, default=5)
    parser.add_argument("--n-wrong", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"]) / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {cfg['model_type']}  |  Device: {device}")
    print(f"Output: {output_dir}\n")

    print("[1/3] Class overview …")
    generate_class_overview(cfg, device, output_dir, seed=args.seed)

    print("[2/3] Correct vs wrong …")
    generate_correct_vs_wrong(
        cfg, device, output_dir,
        n_correct=args.n_correct, n_wrong=args.n_wrong, seed=args.seed,
    )

    if args.compare:
        cfg_comp = load_config(args.compare)
        print("[3/3] Model comparison …")
        generate_model_comparison(cfg, cfg_comp, device, output_dir, seed=args.seed)
    else:
        print("[3/3] Skipped model comparison (no --compare flag)")

    print("\nDone.")


if __name__ == "__main__":
    main()
