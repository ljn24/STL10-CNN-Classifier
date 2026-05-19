"""Plotting utilities for Grad-CAM visualisation."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation and return a ``[H, W, 3]`` uint8 array."""
    img = tensor.cpu().clone()
    img = img * _STD + _MEAN
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def apply_heatmap(
    cam: np.ndarray,
    original: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a ``[H, W]`` heatmap (0-1) onto an ``[H, W, 3]`` uint8 image."""
    cmap = cm.get_cmap(colormap)
    heatmap_rgb = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap_rgb + (1 - alpha) * original).astype(np.uint8)
    return blended


def _add_row_label(ax, text: str, **kwargs) -> None:
    """Place a label to the left of *ax* using an annotation that won't clip."""
    defaults = dict(
        fontsize=11, ha="right", va="center",
        fontweight="bold", color="black",
    )
    defaults.update(kwargs)
    ax.annotate(
        text,
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-10, 0), textcoords="offset points",
        **defaults,
    )


# ── Report-level figure builders ────────────────────────────────────────


def plot_class_overview(
    rows: list[dict],
    title: str = "Grad-CAM Class Overview",
) -> Figure:
    """Build a 10-row x 3-col grid (original / heatmap / overlay).

    Each element of *rows* is a dict with keys:
        ``class_name``, ``original`` (H,W,3 uint8), ``cam`` (H,W float),
        ``pred_class``, ``confidence``.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM", "Overlay"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold")

    for i, row in enumerate(rows):
        orig = row["original"]
        cam_map = row["cam"]
        overlay = apply_heatmap(cam_map, orig)
        cam_color = (cm.get_cmap("jet")(cam_map)[:, :, :3] * 255).astype(np.uint8)

        for j, img in enumerate([orig, cam_color, overlay]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

        _add_row_label(axes[i, 0], row["class_name"])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.14, top=0.95, wspace=0.05, hspace=0.15)
    return fig


def plot_model_comparison(
    rows: list[dict],
    title: str = "Baseline vs Advanced — Grad-CAM Comparison",
) -> Figure:
    """Build a comparison grid: original | baseline overlay | advanced overlay.

    Each element of *rows* is a dict with keys:
        ``class_name``, ``original`` (H,W,3), ``cam_baseline`` (H,W),
        ``cam_advanced`` (H,W).
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Baseline", "Advanced"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold")

    for i, row in enumerate(rows):
        orig = row["original"]
        overlay_bl = apply_heatmap(row["cam_baseline"], orig)
        overlay_adv = apply_heatmap(row["cam_advanced"], orig)

        for j, img in enumerate([orig, overlay_bl, overlay_adv]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

        _add_row_label(axes[i, 0], row["class_name"])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.14, top=0.95, wspace=0.05, hspace=0.15)
    return fig


def plot_correct_vs_wrong(
    correct_rows: list[dict],
    wrong_rows: list[dict],
    class_names: list[str],
    title: str = "Correct vs Wrong Predictions",
) -> Figure:
    """Show correctly-classified samples on top, misclassified on bottom.

    Each row dict has: ``original``, ``cam``, ``true_label`` (int),
    ``pred_label`` (int), ``confidence``.
    """
    n_correct = len(correct_rows)
    n_wrong = len(wrong_rows)
    n_total = n_correct + n_wrong
    if n_total == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(n_total, 2, figsize=(8, 2.8 * n_total))
    if n_total == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Grad-CAM Overlay", fontsize=12, fontweight="bold")

    def _draw(idx: int, row: dict, is_correct: bool) -> None:
        orig = row["original"]
        overlay = apply_heatmap(row["cam"], orig)
        axes[idx, 0].imshow(orig)
        axes[idx, 1].imshow(overlay)
        for j in range(2):
            axes[idx, j].axis("off")

        true_name = class_names[row["true_label"]]
        pred_name = class_names[row["pred_label"]]
        conf = row["confidence"]
        if is_correct:
            label = f"{pred_name} ({conf:.0%})"
            color = "green"
        else:
            label = f"{true_name} -> {pred_name} ({conf:.0%})"
            color = "red"
        _add_row_label(axes[idx, 0], label, color=color, fontweight="normal")

    for i, row in enumerate(correct_rows):
        _draw(i, row, is_correct=True)

    if n_correct > 0 and n_wrong > 0:
        for j in range(2):
            axes[n_correct, j].set_title("")

    for i, row in enumerate(wrong_rows):
        _draw(n_correct + i, row, is_correct=False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.22, top=0.95, wspace=0.05, hspace=0.15)
    return fig
