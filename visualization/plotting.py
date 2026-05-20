"""Plotting utilities for Grad-CAM visualisation — report-ready horizontal layouts."""

from __future__ import annotations

import math

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)

NCOLS = 5
_FIG_WIDTH = 10


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
    return (alpha * heatmap_rgb + (1 - alpha) * original).astype(np.uint8)


def _row_label(ax, text: str, **kwargs) -> None:
    """Place a label to the left of *ax*."""
    defaults = dict(fontsize=10, ha="right", va="center", fontweight="bold", color="black")
    defaults.update(kwargs)
    ax.annotate(
        text, xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-8, 0), textcoords="offset points", **defaults,
    )


def _build_ratios(ngroups: int, rpg: int, gap: float = 0.3) -> list[float]:
    """Height ratios for *ngroups* row-groups with *rpg* content rows each."""
    if ngroups == 0:
        return []
    ratios: list[float] = []
    for g in range(ngroups):
        ratios.extend([1.0] * rpg)
        if g < ngroups - 1:
            ratios.append(gap)
    return ratios


def _figure_grid(ratios: list[float]) -> tuple[Figure, GridSpec]:
    fig_h = sum(r * 1.7 for r in ratios) + 1.0
    fig = plt.figure(figsize=(_FIG_WIDTH, fig_h))
    gs = GridSpec(
        len(ratios), NCOLS, figure=fig,
        height_ratios=ratios, hspace=0.06, wspace=0.02,
    )
    return fig, gs


def _group_row_base(group: int, rows_per_group: int) -> int:
    return group * (rows_per_group + 1)


# ── Report-level figure builders ────────────────────────────────────────


def plot_class_overview(
    rows: list[dict],
    title: str = "Grad-CAM Class Overview",
) -> Figure:
    """Horizontal grid: 5 classes per row-group, 2 sub-rows (original + overlay).

    Each element of *rows*: ``class_name``, ``original`` (H,W,3), ``cam`` (H,W).
    """
    rpg = 2
    ngroups = math.ceil(len(rows) / NCOLS)
    fig, gs = _figure_grid(_build_ratios(ngroups, rpg))

    group_leaders: list[tuple[plt.Axes, plt.Axes]] = []
    for i, row in enumerate(rows):
        g, c = divmod(i, NCOLS)
        rb = _group_row_base(g, rpg)
        orig = row["original"]
        overlay = apply_heatmap(row["cam"], orig)

        ax0 = fig.add_subplot(gs[rb, c])
        ax0.imshow(orig)
        ax0.axis("off")
        ax0.set_title(row["class_name"], fontsize=10, fontweight="bold")

        ax1 = fig.add_subplot(gs[rb + 1, c])
        ax1.imshow(overlay)
        ax1.axis("off")

        if c == 0:
            group_leaders.append((ax0, ax1))

    for ax0, ax1 in group_leaders:
        _row_label(ax0, "Original")
        _row_label(ax1, "Overlay")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.01)
    return fig


def plot_model_comparison(
    rows: list[dict],
    title: str = "Baseline vs Advanced \u2014 Grad-CAM",
) -> Figure:
    """Horizontal grid: 5 classes per group, 3 sub-rows (original / baseline / advanced).

    Each element of *rows*: ``class_name``, ``original``, ``cam_baseline``, ``cam_advanced``.
    """
    rpg = 3
    ngroups = math.ceil(len(rows) / NCOLS)
    fig, gs = _figure_grid(_build_ratios(ngroups, rpg, gap=0.35))

    group_leaders: list[list[plt.Axes]] = []
    for i, row in enumerate(rows):
        g, c = divmod(i, NCOLS)
        rb = _group_row_base(g, rpg)
        orig = row["original"]
        images = [
            orig,
            apply_heatmap(row["cam_baseline"], orig),
            apply_heatmap(row["cam_advanced"], orig),
        ]

        col_axes: list[plt.Axes] = []
        for s, img in enumerate(images):
            ax = fig.add_subplot(gs[rb + s, c])
            ax.imshow(img)
            ax.axis("off")
            col_axes.append(ax)

        col_axes[0].set_title(row["class_name"], fontsize=10, fontweight="bold")
        if c == 0:
            group_leaders.append(col_axes)

    sub_labels = ["Original", "Baseline", "Advanced"]
    for col_axes in group_leaders:
        for ax, lb in zip(col_axes, sub_labels):
            _row_label(ax, lb)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.94, bottom=0.01)
    return fig


def plot_correct_vs_wrong(
    correct_rows: list[dict],
    wrong_rows: list[dict],
    class_names: list[str],
    title: str = "Correct vs Wrong Predictions",
) -> Figure:
    """Two horizontal sections (correct / wrong), each with original + overlay rows.

    Each row dict: ``original``, ``cam``, ``true_label``, ``pred_label``, ``confidence``.
    """
    rpg = 2
    nc, nw = len(correct_rows), len(wrong_rows)
    gc = math.ceil(nc / NCOLS) if nc else 0
    gw = math.ceil(nw / NCOLS) if nw else 0

    if nc + nw == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.axis("off")
        return fig

    ratios = _build_ratios(gc, rpg)
    if gc > 0 and gw > 0:
        ratios.append(0.45)
    ratios.extend(_build_ratios(gw, rpg))

    fig, gs = _figure_grid(ratios)

    correct_rows_total = gc * rpg + max(0, gc - 1)
    wrong_offset = correct_rows_total + (1 if gc > 0 and gw > 0 else 0)

    first_correct: tuple[plt.Axes, plt.Axes] | None = None
    first_wrong: tuple[plt.Axes, plt.Axes] | None = None

    for i, row in enumerate(correct_rows):
        g, c = divmod(i, NCOLS)
        rb = _group_row_base(g, rpg)
        orig = row["original"]
        overlay = apply_heatmap(row["cam"], orig)
        pred = class_names[row["pred_label"]]
        conf = row["confidence"]

        ax0 = fig.add_subplot(gs[rb, c])
        ax0.imshow(orig)
        ax0.axis("off")
        ax0.set_title(f"{pred} ({conf:.0%})", fontsize=9, color="green", fontweight="bold")

        ax1 = fig.add_subplot(gs[rb + 1, c])
        ax1.imshow(overlay)
        ax1.axis("off")

        if first_correct is None:
            first_correct = (ax0, ax1)

    for i, row in enumerate(wrong_rows):
        g, c = divmod(i, NCOLS)
        rb = wrong_offset + _group_row_base(g, rpg)
        orig = row["original"]
        overlay = apply_heatmap(row["cam"], orig)
        true_n = class_names[row["true_label"]]
        pred_n = class_names[row["pred_label"]]
        conf = row["confidence"]

        ax0 = fig.add_subplot(gs[rb, c])
        ax0.imshow(orig)
        ax0.axis("off")
        ax0.set_title(
            f"{true_n} \u2192 {pred_n} ({conf:.0%})",
            fontsize=9, color="red", fontweight="bold",
        )

        ax1 = fig.add_subplot(gs[rb + 1, c])
        ax1.imshow(overlay)
        ax1.axis("off")

        if first_wrong is None:
            first_wrong = (ax0, ax1)

    if gc > 0 and gw > 0:
        div_ax = fig.add_subplot(gs[correct_rows_total, :])
        div_ax.axis("off")
        div_ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")

    if first_correct is not None:
        _row_label(first_correct[0], "Original")
        _row_label(first_correct[1], "Overlay")
    if first_wrong is not None:
        _row_label(first_wrong[0], "Original")
        _row_label(first_wrong[1], "Overlay")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.01)
    return fig
