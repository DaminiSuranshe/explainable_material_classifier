"""
Validates feedback data before retraining.
"""

from typing import Dict
from pathlib import Path

from src.data_management.category_manager import CategoryManager
from src.utils.logger import setup_logger

logger = setup_logger("feedback_validator")


class FeedbackValidator:
    """
    Ensures feedback correctness and safety.
    """

    def __init__(self, category_file: str):
        self.category_manager = CategoryManager(category_file)

    def validate(self, record: Dict) -> bool:
        """
        Validate feedback record.

        Args:
            record: Feedback entry

        Returns:
            True if valid
        """
        try:
            major = record["correct_major"]
            sub = record["correct_sub"]
            image_path = Path(record["image_path"])

            self.category_manager.encode(major, sub)

            if not image_path.exists():
                logger.warning("Image missing: %s", image_path)
                return False

            return True

        except Exception as e:
            logger.error("Invalid feedback: %s", e)
            return False
