"""
Admin endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/model/info")
async def model_info():
    return {
        "model": "Hierarchical ResNet50",
        "version": "v1.0.0",
        "framework": "PyTorch",
        "explainability": ["Grad-CAM", "CV Feature Analysis"],
    }

@router.post("/retrain")
async def retrain_model():
    """
    Trigger retraining manually.
    """
    return {
        "status": "retraining scheduled",
        "note": "Run retrain_scheduler with validated feedback data",
    }

@router.get("/metrics")
async def metrics():
    return {
        "requests_per_minute_limit": 10,
        "model_loaded": True,
        "cache_enabled": True,
    }
