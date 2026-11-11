"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__
from app.schemas import HealthResponse
from models.inference import get_model

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    model_loaded = False
    try:
        get_model(settings.model_path)
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=str(settings.model_path),
        version=__version__,
    )
