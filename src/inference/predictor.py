"""
Single-image inference pipeline.
"""

from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from ..models.cnn_classifier import HierarchicalCNN
from ..preprocessing.image_processor import ImageProcessor
from ..data_management.category_manager import CategoryManager
from src.inference.object_detector import detect_multiple_objects


class Predictor:
    """
    Loads trained model and runs inference.
    API-safe: starts even if model file is missing.
    """

    def __init__(
        self,
        model_path: str | Path | None,
        category_file: str | Path,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # --- Load category metadata ---
        self.category_manager = CategoryManager(category_file)
        num_major, num_sub = self.category_manager.num_classes()

        # --- Initialize model architecture ---
        self.model = HierarchicalCNN(num_major, num_sub)

        # --- Resolve model path ---
        if model_path is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = (
                project_root
                / "models"
                / "saved_models"
                / "hierarchical_cnn_best.pth"
            )
        else:
            model_path = Path(model_path)

        # --- Load model if available ---
        self.model_loaded = model_path.exists()

        if self.model_loaded:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded from: {model_path}")
        else:
            print(
                "⚠️ Model file not found. "
                "API is running, but predictions are disabled."
            )

        # --- Image preprocessing ---
        self.processor = ImageProcessor()

    def predict(self, image_path: str | Path) -> Dict:
        """
        Run inference on a single image.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Train the model or provide "
                "models/saved_models/hierarchical_cnn_best.pth"
            )

        image_path = Path(image_path)

        image = Image.open(image_path).convert("RGB")
        tensor = self.processor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        major_probs = torch.softmax(outputs["major_logits"], dim=1)
        sub_probs = torch.softmax(outputs["sub_logits"], dim=1)

        major_id = major_probs.argmax(dim=1).item()
        sub_id = sub_probs.argmax(dim=1).item()

        major, sub = self.category_manager.decode(major_id, sub_id)

        return {
            "major_category": major,
            "subcategory": sub,
            "confidence": {
                "major": float(major_probs.max().item()),
                "sub": float(sub_probs.max().item()),
            },
        }

def predict(self, image_path: str):
    image = cv2.imread(image_path)

    detection_result = detect_multiple_objects(image)

    predictions = []

    for obj in detection_result["objects"]:
        crop = obj["crop"]
        # Existing single-object pipeline runs here
        prediction = self._predict_single_crop(crop)

        predictions.append({
            "object_id": obj["object_id"],
            "bounding_box": obj["bounding_box"],
            "prediction": prediction,
        })

    return predictions
