"""
Explainable prediction pipeline.
"""

import cv2
import numpy as np
import torch
from typing import Dict

from src.inference.predictor import Predictor
from src.explainability.gradcam import GradCAM
from src.explainability.feature_extractors.texture_analyzer import TextureAnalyzer
from src.explainability.feature_extractors.shape_analyzer import ShapeAnalyzer
from src.explainability.feature_extractors.reflectivity import ReflectivityEstimator
from src.explainability.feature_extractors.transparency import TransparencyDetector
from src.explainability.explanation_engine import ExplanationGenerator


class ExplainablePredictor(Predictor):
    """
    Predictor with explainability support.
    """

    def explain(self, image_path: str) -> Dict:
        prediction = self.predict(image_path)

        image_bgr = cv2.imread(image_path)

        texture = TextureAnalyzer().analyze(image_bgr)
        shape = ShapeAnalyzer().analyze(image_bgr)
        reflectivity = ReflectivityEstimator().estimate(image_bgr)
        transparency = TransparencyDetector().detect(image_bgr)

        features = {
            **texture,
            **shape,
            "reflectivity": reflectivity,
            "transparency": transparency,
        }

        explanation = ExplanationGenerator().generate(
            features,
            prediction["major_category"],
            prediction["subcategory"],
        )

        return {
            **prediction,
            "explanation": explanation,
            "attention_map_available": True,
        }
        