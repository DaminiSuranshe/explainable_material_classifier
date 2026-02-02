"""
Pydantic schemas for API validation.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class ConfidenceScores(BaseModel):
    major_category: float
    subcategory: float


class AlternativePrediction(BaseModel):
    category: str
    subcategory: str
    confidence: float


class Explanation(BaseModel):
    key_visual_features: List[str]
    feature_confidence: Dict[str, float]
    reasoning_summary: str
    attention_map_available: bool


class PredictionResponse(BaseModel):
    predicted_category: str
    predicted_subcategory: str
    confidence_scores: ConfidenceScores
    explanation: Explanation
    processing_time_ms: int
    model_version: str
    timestamp: str


class FeedbackRequest(BaseModel):
    image_id: str
    correct_major: str
    correct_sub: str
