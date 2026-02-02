"""
Image preprocessing and normalization.
"""

from typing import Tuple
import torch
from torchvision import transforms
from PIL import Image


class ImageProcessor:
    """
    Image preprocessing pipeline.
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
