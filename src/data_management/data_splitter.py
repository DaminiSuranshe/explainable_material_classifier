"""
Dataset splitting utility.
"""

import random
import shutil
from pathlib import Path
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger("data_splitter")


def split_dataset(
    raw_dir: str,
    output_dir: str,
    split_ratio: Tuple[float, float, float],
    seed: int = 42,
) -> None:
    """
    Split dataset into train/validation/test.

    Args:
        raw_dir: Raw dataset directory
        output_dir: Output directory for splits
        split_ratio: (train, val, test)
        seed: Random seed
    """
    random.seed(seed)

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if sum(split_ratio) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    for major_dir in raw_dir.iterdir():
        if not major_dir.is_dir():
            continue

        for sub_dir in major_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            images = list(sub_dir.glob("*.*"))
            random.shuffle(images)

            n = len(images)
            n_train = int(n * split_ratio[0])
            n_val = int(n * split_ratio[1])

            splits = {
                "train": images[:n_train],
                "validation": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }

            for split, files in splits.items():
                target_dir = output_dir / split / major_dir.name / sub_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                for img in files:
                    shutil.copy(img, target_dir / img.name)

    logger.info("Dataset split complete")
