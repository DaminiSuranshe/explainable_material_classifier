"""
Feedback collection and storage.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger("feedback_handler")


class FeedbackHandler:
    """
    Stores validated user feedback in versioned format.
    """

    def __init__(self, feedback_dir: str = "dataset/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_file = self.feedback_dir / "feedback.json"

    def save_feedback(self, record: Dict) -> None:
        """
        Append feedback record.

        Args:
            record: Feedback dictionary
        """
        record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        feedback: List[Dict] = []
        if self.feedback_file.exists():
            feedback = json.loads(self.feedback_file.read_text())

        feedback.append(record)
        self.feedback_file.write_text(json.dumps(feedback, indent=2))

        logger.info("Feedback recorded (%d total)", len(feedback))
