"""
FastAPI application entrypoint.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.middleware import rate_limiter

from ..inference.predictor import Predictor
from .routes import prediction, feedback, categories, admin

BASE_DIR = Path(__file__).resolve().parents[2]
CATEGORY_FILE = BASE_DIR / "config" / "categories.json"

app = FastAPI(
    title="Explainable Material Classification API",
    version="1.0.0",
)

@app.on_event("startup")
def startup():
    app.state.predictor = Predictor(
        model_path=None,              # ✅ CRITICAL FIX
        category_file=CATEGORY_FILE,  # ✅ correct
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
app.include_router(feedback.router)
app.include_router(categories.router)
app.include_router(admin.router)
app.middleware("http")(rate_limiter)

@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.predictor.model_loaded
    }
