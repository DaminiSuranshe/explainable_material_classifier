"""
Category hierarchy manager.

- Loads hierarchical categories from JSON
- Dynamically assigns IDs
- Supports adding new categories without code changes
"""

from typing import Dict, List, Tuple
import json
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("category_manager")


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

    def decode(self, major_id: int, sub_id: int) -> Tuple[str, str]:
        """
        Decode numeric IDs back to category names.

        Args:
            major_id: Major category ID
            sub_id: Subcategory ID

        Returns:
            (major_name, sub_name)
        """
        major = self.id_to_major[major_id]
        sub_full = self.id_to_sub[sub_id]
        _, sub = sub_full.split("::")

        return major, sub

    def num_classes(self) -> Tuple[int, int]:
        """
        Returns:
            (num_major_classes, num_sub_classes)
        """
        return len(self.major_to_id), len(self.sub_to_id)
