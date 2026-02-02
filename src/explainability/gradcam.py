"""
Grad-CAM implementation for CNN explainability.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from src.utils.logger import setup_logger

logger = setup_logger("gradcam")


class GradCAM:
    """
    Generates Grad-CAM heatmaps for CNN models.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Args:
            model: Trained CNN model
            target_layer: Name of convolutional layer
        """
        self.model = model
        self.target_layer = dict(
            [*self.model.named_modules()]
        )[target_layer]

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(
        self, input_tensor: torch.Tensor, class_idx: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor
            class_idx: Target class index

        Returns:
            Heatmap (H x W) normalized
        """
        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output["sub_logits"][:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam
