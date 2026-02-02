"""
Small-data dataset loader with heavy augmentation.

This loader is designed for:
- 10–50 images per class
- Transfer learning
- Aggressive on-the-fly augmentation

It integrates with the existing dataset structure:
dataset/splits/{train,validation,test}/<Major>/<Sub>/*.jpg
"""

from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data_management.category_manager import CategoryManager
from src.preprocessing.heavy_augmentation import HeavyAugmentation


class SmallDataImageDataset(Dataset):
    """
    Dataset wrapper optimized for very small datasets.
    """

    def __init__(
        self,
        root_dir: str,
        category_manager: CategoryManager,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        augmentation_multiplier: int = 25,
    ):
        self.root_dir = Path(root_dir)
        self.category_manager = category_manager
        self.is_training = is_training
        self.augmentation_multiplier = augmentation_multiplier

        self.transform = HeavyAugmentation(
            image_size=image_size,
            is_training=is_training,
        )

        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Counter = Counter()

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """
        Scan dataset and build sample index.
        """
        for major_dir in self.root_dir.iterdir():
            if not major_dir.is_dir():
                continue

            for sub_dir in major_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                major = major_dir.name
                sub = sub_dir.name
                _, sub_id = self.category_manager.encode(major, sub)

                for img_path in sub_dir.iterdir():
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue

                    self.samples.append((img_path, sub_id))
                    self.class_counts[sub_id] += 1

        if len(self.samples) == 0:
            raise RuntimeError("No images found in dataset")

    def __len__(self) -> int:
        """
        Artificially inflate dataset size via augmentation.
        """
        if self.is_training:
            return len(self.samples) * self.augmentation_multiplier
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one augmented sample.
        """
        if self.is_training:
            idx = idx % len(self.samples)

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights.
        """
        total = sum(self.class_counts.values())
        weights = {
            cls: total / count for cls, count in self.class_counts.items()
        }

        max_cls = max(weights.keys())
        weight_tensor = torch.zeros(max_cls + 1)

        for cls, weight in weights.items():
            weight_tensor[cls] = weight

        return weight_tensor
