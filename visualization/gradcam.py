"""Grad-CAM implementation using PyTorch hooks.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Generate Grad-CAM heatmaps for a given model and target layer.

    The target layer should be the last convolutional layer before the
    global average pooling (i.e. ``model.features`` for both BaselineModel
    and AdvancedModel).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.model.eval()

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self, _module: nn.Module, _input: tuple, output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self, _module: nn.Module, _grad_input: tuple, grad_output: tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """Compute a Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Pre-processed image tensor of shape ``[1, C, H, W]``.
            target_class: Class index to visualise. If *None*, the predicted
                class (argmax) is used.

        Returns:
            ``(cam, predicted_class, confidence)`` where *cam* is a
            ``[H, W]`` tensor in ``[0, 1]`` (same spatial size as input).
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        pred_class = logits.argmax(dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        score = logits[0, target_class]
        score.backward()

        # alpha_k = GAP of gradients  →  [C]
        weights = self._gradients[0].mean(dim=(1, 2))

        # Weighted combination of activation maps  →  [H_feat, W_feat]
        cam = (weights[:, None, None] * self._activations[0]).sum(dim=0)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(h, w),
            mode="bilinear", align_corners=False,
        ).squeeze()

        return cam, pred_class, confidence

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()
