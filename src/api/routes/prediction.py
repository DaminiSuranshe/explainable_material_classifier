"""
Prediction endpoint.
"""

import time
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Request

router = APIRouter()


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    start_time = time.time()

    # --- Get shared predictor from app state ---
    predictor = request.app.state.predictor

    # --- Guard: model not loaded ---
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train the model or add "
                "models/saved_models/hierarchical_cnn_best.pth"
            ),
        )

    # --- Validate image type ---
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # --- Save uploaded image ---
    image_dir = Path("dataset/raw/uploads")
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --- Run prediction + explanation ---
    result = predictor.explain(str(image_path))

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "predicted_category": result["major_category"],
        "predicted_subcategory": result["subcategory"],
        "confidence_scores": {
            "major_category": result["confidence"]["major"],
            "subcategory": result["confidence"]["sub"],
        },
        "explanation": {
            **result.get("explanation", {}),
            "attention_map_available": True,
        },
        "processing_time_ms": elapsed_ms,
        "model_version": "v1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@router.post("/predict/batch")
async def batch_predict(files: list[UploadFile]):
    results = []
    for file in files:
        result = await predict(file)
        results.append(result)
    return results
