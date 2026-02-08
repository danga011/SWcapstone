#!/usr/bin/env python3
"""
FastAPI server for anomaly detection inference.

Usage:
    uvicorn src.serve:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict - Predict anomaly for a single image
    GET /health - Health check
"""

import io
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# Lazy imports for models
_model = None
_model_type = None
_device = None


def _detect_model_type(data: dict) -> str:
    """Auto-detect model type from pickle contents."""
    if "memory_bank" in data:
        return "patchcore"
    elif "cov_inv" in data:
        return "padim"
    else:
        raise ValueError(
            f"Cannot detect model type from pickle keys: {list(data.keys())}. "
            "Set MODEL_TYPE env var to 'padim' or 'patchcore'."
        )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    heatmap_max: float
    heatmap_mean: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_type: str
    device: str


app = FastAPI(
    title="Anomaly Detection API",
    description="PaDiM/PatchCore anomaly detection service (auto-detects model type)",
    version="1.1.0",
)


def get_device():
    """Get the device for inference."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_model():
    """Load the anomaly detection model (lazy loading, supports PaDiM and PatchCore)."""
    global _model, _model_type

    if _model is not None:
        return _model

    model_path = os.environ.get(
        "MODEL_PATH", "models/patchcore_mvtec_coreset010/heatmap_model.pkl"
    )

    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found at {model_path}. Run training first.")

    # Load pickle to inspect contents
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    # Determine model type
    env_model_type = os.environ.get("MODEL_TYPE", "auto").lower()
    if env_model_type == "auto":
        _model_type = _detect_model_type(data)
    elif env_model_type in ("padim", "patchcore"):
        _model_type = env_model_type
    else:
        raise ValueError(
            f"Invalid MODEL_TYPE='{env_model_type}'. Use 'auto', 'padim', or 'patchcore'."
        )

    device_str = str(get_device())

    if _model_type == "patchcore":
        from src.models.heatmap import PatchCore

        _model = PatchCore(
            backbone=data.get("backbone", "resnet18"),
            layers=data.get("layers", ["layer2", "layer3"]),
            image_size=data.get("image_size", 224),
            coreset_sampling_ratio=data.get("coreset_sampling_ratio", 0.1),
            num_neighbors=data.get("num_neighbors", 9),
            device=device_str,
        )
    elif _model_type == "padim":
        from src.models.heatmap import PaDiM

        _model = PaDiM(
            backbone=data.get("backbone", "resnet18"),
            layers=data.get("layers", ["layer1", "layer2", "layer3"]),
            image_size=data.get("image_size", 224),
            device=device_str,
        )

    _model.load(model_path)
    return _model


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_type=_model_type or "not_loaded",
        device=str(get_device()),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5,
):
    """
    Predict anomaly for an uploaded image.

    Args:
        file: Image file (JPEG, PNG)
        threshold: Anomaly score threshold (default: 0.5)

    Returns:
        Prediction results including anomaly score and heatmap statistics
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Get model
        model = get_model()

        # Preprocess
        input_tensor = preprocess_image(image, model.image_size)

        # Predict
        scores, heatmaps = model.predict(input_tensor)
        anomaly_score = float(scores[0])
        heatmap = heatmaps[0]

        # Normalize score to 0-1 range for confidence
        confidence = min(max(anomaly_score / 100.0, 0.0), 1.0)

        return PredictionResponse(
            is_anomaly=anomaly_score > threshold,
            anomaly_score=anomaly_score,
            confidence=confidence,
            heatmap_max=float(np.max(heatmap)),
            heatmap_mean=float(np.mean(heatmap)),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_model()
        print(f"Model loaded successfully (type: {_model_type})")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
