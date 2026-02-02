"""
Heavy augmentation pipeline for small datasets.

Purpose:
- Expand 10–15 images into hundreds of diverse samples
- Prevent memorization
- Improve generalization with limited data

This module is SAFE for explainability:
- No label-altering transforms
- No irreversible distortions
"""

from typing import Tuple
import random
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


class HeavyAugmentation:
    """
    Aggressive augmentation pipeline for few-shot learning.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
    ):
        self.image_size = image_size
        self.is_training = is_training

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.8, 1.2)
                ),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.25,
                    hue=0.05,
                ),
                transforms.GaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(0.1, 2.0),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                RandomCutout(p=0.5, size_ratio=0.3),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.is_training:
            return self.train_transform(image)
        return self.val_transform(image)

    @staticmethod
    def preview(
        image_path: str,
        num_samples: int = 25,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """
        Visualize augmentation diversity from a single image.

        Usage:
            HeavyAugmentation.preview("sample.jpg", 25)
        """
        image = Image.open(image_path).convert("RGB")
        augmenter = HeavyAugmentation(
            image_size=image_size, is_training=True
        )

        cols = 5
        rows = num_samples // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

        for i in range(num_samples):
            aug_img = augmenter(image)
            aug_img = aug_img.permute(1, 2, 0).numpy()
            aug_img = (aug_img * 0.225 + 0.45).clip(0, 1)

            ax = axes[i // cols, i % cols]
            ax.imshow(aug_img)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


class RandomCutout:
    """
    Randomly masks a rectangular region in the image.
    """

    def __init__(self, p: float = 0.5, size_ratio: float = 0.3):
        self.p = p
        self.size_ratio = size_ratio

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        _, h, w = tensor.shape
        cutout_size = int(min(h, w) * self.size_ratio)

        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)

        tensor[:, y : y + cutout_size, x : x + cutout_size] = 0
        return tensor
