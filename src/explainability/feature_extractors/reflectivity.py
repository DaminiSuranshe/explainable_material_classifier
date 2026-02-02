"""
Surface reflectivity estimation.
"""

import cv2
import numpy as np


class ReflectivityEstimator:
    """
    Estimates specular reflectivity from highlights.
    """

    def estimate(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = gray > 230

        reflectivity = bright_pixels.sum() / gray.size
        return round(float(reflectivity), 3)
