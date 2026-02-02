"""
Hierarchical dataset loader for material classification.

Expected folder structure:
dataset/splits/train/<Major>/<Sub>/<image>.jpg
"""

from typing import List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.data_management.category_manager import CategoryManager
from src.preprocessing.image_processor import ImageProcessor
from src.utils.logger import setup_logger

logger = setup_logger("dataset_loader")


class HierarchicalImageDataset(Dataset):
    """
    PyTorch Dataset for hierarchical image classification.
    """

    def __init__(
        self,
        root_dir: str,
        category_manager: CategoryManager,
        transform: ImageProcessor,
    ):
        """
        Args:
            root_dir: Dataset split directory
            category_manager: CategoryManager instance
            transform: ImageProcessor instance
        """
        self.root_dir = Path(root_dir)
        self.category_manager = category_manager
        self.transform = transform

        self.samples: List[Tuple[Path, int, int]] = []

        self._scan_dataset()

    def _scan_dataset(self) -> None:
        """
        Recursively scan dataset directory.
        """
        logger.info("Scanning dataset at %s", self.root_dir)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for major_dir in self.root_dir.iterdir():
            if not major_dir.is_dir():
                continue

            for sub_dir in major_dir.iterdir():
                if not sub_dir.is_dir():
                    continue

                for img_path in sub_dir.iterdir():
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue

                    major = major_dir.name
                    sub = sub_dir.name

                    major_id, sub_id = self.category_manager.encode(major, sub)
                    self.samples.append((img_path, major_id, sub_id))

        logger.info("Loaded %d samples", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError("No images found in dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, major_id, sub_id = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image %s: %s", img_path, e)
            raise

        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "major_label": torch.tensor(major_id, dtype=torch.long),
            "sub_label": torch.tensor(sub_id, dtype=torch.long),
        }
