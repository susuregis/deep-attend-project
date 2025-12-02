from datetime import datetime

from fastapi import APIRouter, Request

from core.config import get_model_config


router = APIRouter(tags=["Status"])
MODEL_CONFIG = get_model_config()


@router.get("/")
def root(request: Request):
    request_count = getattr(request.app.state, "request_count", 0)
    device = getattr(request.app.state, "device", "cpu")

    return {
        "message": "API de Detecção de Atenção",
        "status": "online",
        "model": MODEL_CONFIG["model_type"],
        "accuracy": MODEL_CONFIG["accuracy"],
        "classes": MODEL_CONFIG["classes"],
        "device": device,
        "requests_processed": request_count,
        "version": "2.0.0",
    }


@router.get("/health")
def health(request: Request):
    device = getattr(request.app.state, "device", "cpu")
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": True,
        "timestamp": datetime.now().isoformat(),
    }

