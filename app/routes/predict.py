"""Win-probability prediction route."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.schemas import PredictResponse, QuoteRequest
from models.inference import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: QuoteRequest, request: Request) -> PredictResponse:
    started = time.perf_counter()
    settings = request.app.state.settings
    try:
        model = get_model(settings.model_path)
        win_probability = model.predict_win_probability(payload)
        recommended_price = model.recommend_price(payload, win_probability=win_probability)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "endpoint=predict latency_ms=%.2f win_probability=%.4f category=%s tier=%s",
        latency_ms,
        win_probability,
        payload.part_category.value,
        payload.dealer_tier.value,
    )
    return PredictResponse(
        win_probability=win_probability,
        recommended_price=recommended_price,
        latency_ms=latency_ms,
        model_name=model.model_name,
    )
