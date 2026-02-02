"""
Texture feature extraction.
"""

import cv2
import numpy as np
from typing import Dict


class TextureAnalyzer:
    """
    Analyzes surface texture smoothness and uniformity.
    """

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        smoothness = float(np.exp(-variance / 1000))
        uniformity = float(1.0 - np.std(gray) / 128)

        return {
            "smoothness": round(smoothness, 3),
            "uniformity": round(uniformity, 3),
        }
