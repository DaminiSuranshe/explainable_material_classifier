"""
Shape and geometry analysis.
"""

import cv2
import numpy as np
from typing import Dict


class ShapeAnalyzer:
    """
    Detects geometric regularity and shape consistency.
    """

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"shape_regularity": 0.0}

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        circularity = (
            4 * np.pi * area / (perimeter ** 2 + 1e-8)
        )

        return {
            "shape_regularity": round(float(circularity), 3)
        }
