"""
Transparency estimation.
"""

import cv2
import numpy as np


class TransparencyDetector:
    """
    Estimates transparency using edge contrast and clarity.
    """

    def detect(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_density = edges.sum() / edges.size
        transparency = float(1.0 - edge_density)

        return round(transparency, 3)
