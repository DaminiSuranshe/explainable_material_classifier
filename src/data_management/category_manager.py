"""
Category hierarchy manager.

- Loads hierarchical categories from JSON
- Dynamically assigns IDs
- Supports encoding and decoding
- Stable across training and inference
"""

from typing import Dict, Tuple
import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("category_manager")


class CategoryManager:
    """
    Manages hierarchical material categories and label encoding.
    """

    def __init__(self, category_file: str):
        """
        Args:
            category_file: Path to categories.json
        """
        self.category_file = Path(category_file)

        self.major_to_id: Dict[str, int] = {}
        self.sub_to_id: Dict[str, int] = {}

        self.id_to_major: Dict[int, str] = {}
        self.id_to_sub: Dict[int, str] = {}

        self._load_categories()

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def num_subcategories(self) -> int:
        return len(self.sub_to_id)

    @property
    def num_major_categories(self) -> int:
        return len(self.major_to_id)

    # --------------------------------------------------
    # Loading
    # --------------------------------------------------

    def _load_categories(self) -> None:
        """
        Load and parse category hierarchy from JSON.
        """
        logger.info("Loading categories from %s", self.category_file)

        if not self.category_file.exists():
            raise FileNotFoundError(f"Category file not found: {self.category_file}")

        with open(self.category_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        major_categories = data["categories"]

        major_id = 0
        sub_id = 0

        for major, subs in major_categories.items():
            self.major_to_id[major] = major_id
            self.id_to_major[major_id] = major
            major_id += 1

            for sub in subs.keys():
                full_sub = f"{major}::{sub}"
                self.sub_to_id[full_sub] = sub_id
                self.id_to_sub[sub_id] = full_sub
                sub_id += 1

        logger.info(
            "Loaded %d major categories and %d subcategories",
            len(self.major_to_id),
            len(self.sub_to_id),
        )

    # --------------------------------------------------
    # Encoding
    # --------------------------------------------------

    def encode(self, major: str, sub: str) -> Tuple[int, int]:
        """
        Encode category names to numeric IDs.

        Args:
            major: Major category name
            sub: Subcategory name

        Returns:
            (major_id, sub_id)
        """
        full_sub = f"{major}::{sub}"

        if major not in self.major_to_id:
            raise ValueError(f"Unknown major category: {major}")

        if full_sub not in self.sub_to_id:
            raise ValueError(f"Unknown subcategory: {full_sub}")

        return self.major_to_id[major], self.sub_to_id[full_sub]

    # --------------------------------------------------
    # Decoding
    # --------------------------------------------------

    def decode(self, major_id: int, sub_id: int) -> Tuple[str, str]:
        """
        Decode numeric IDs back to category names.

        Returns:
            (major_name, sub_name)
        """
        if major_id not in self.id_to_major:
            raise ValueError(f"Unknown major id: {major_id}")

        if sub_id not in self.id_to_sub:
            raise ValueError(f"Unknown subcategory id: {sub_id}")

        major = self.id_to_major[major_id]
        _, sub = self.id_to_sub[sub_id].split("::")

        return major, sub

    def decode_subcategory(self, sub_id: int) -> str:
        """
        Decode subcategory ID back to 'Major::Sub' label.
        """
        if sub_id not in self.id_to_sub:
            raise ValueError(f"Unknown subcategory id: {sub_id}")

        return self.id_to_sub[sub_id]

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def num_classes(self) -> Tuple[int, int]:
        """
        Returns:
            (num_major_classes, num_sub_classes)
        """
        return self.num_major_categories, self.num_subcategories
