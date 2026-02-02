"""
User feedback endpoint.
"""

import json
from pathlib import Path
from fastapi import APIRouter
from src.api.schemas import FeedbackRequest

router = APIRouter()

feedback_file = Path("dataset/feedback/feedback.json")
feedback_file.parent.mkdir(parents=True, exist_ok=True)


@router.post("/feedback")
async def submit_feedback(data: FeedbackRequest):
    feedback = []
    if feedback_file.exists():
        feedback = json.loads(feedback_file.read_text())

    feedback.append(data.dict())
    feedback_file.write_text(json.dumps(feedback, indent=2))

    return {"status": "feedback recorded"}
