"""
Explanation generator.
"""

from typing import Dict, List


class ExplanationGenerator:
    """
    Converts visual features into human-readable explanations.
    """

    def generate(
        self,
        features: Dict[str, float],
        major: str,
        sub: str,
    ) -> Dict:
        descriptors: List[str] = []

        if features.get("smoothness", 0) > 0.7:
            descriptors.append("smooth surface texture")

        if features.get("reflectivity", 0) > 0.05:
            descriptors.append("shiny or reflective surface")

        if features.get("shape_regularity", 0) > 0.6:
            descriptors.append("regular geometric shape")

        if features.get("transparency", 0) > 0.6:
            descriptors.append("partially transparent material")

        reasoning = (
            f"The object was classified as {sub} under {major} "
            f"because it exhibits " + ", ".join(descriptors) + "."
        )

        return {
            "key_visual_features": descriptors,
            "feature_confidence": features,
            "reasoning_summary": reasoning,
        }
