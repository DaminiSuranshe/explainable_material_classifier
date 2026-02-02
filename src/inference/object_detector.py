"""
PHASE 6: Multi-object detection integration point.

⚠️ DO NOT IMPLEMENT DETECTION HERE.
This file defines the interface only.
"""

from typing import List, Dict
import numpy as np


def detect_multiple_objects(image: np.ndarray) -> Dict:
    """
    Detect multiple objects in an image.

    CURRENT BEHAVIOR (Phase 6):
        - Returns a single object covering the full image.

    FUTURE BEHAVIOR (Phase 6+):
        - Use YOLO / SSD / Faster R-CNN
        - Return one entry per detected object

    Args:
        image: Input image (H x W x C)

    Returns:
        Dictionary with detected objects and metadata
    """

    height, width, _ = image.shape

    return {
        "objects": [
            {
                "object_id": "object_0",
                "bounding_box": {
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": width,
                    "y_max": height,
                    "confidence": 1.0,
                },
                "crop": image,
            }
        ],
        "image_width": width,
        "image_height": height,
    }
