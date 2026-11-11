"""Price optimization route with commercial guardrails."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.schemas import OptimizePriceRequest, OptimizePriceResponse
from guardrails.clamping import clamp_price
from models.inference import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["optimize"])


@router.post("/optimize-price", response_model=OptimizePriceResponse)
def optimize_price(payload: OptimizePriceRequest, request: Request) -> OptimizePriceResponse:
    started = time.perf_counter()
    settings = request.app.state.settings

    try:
        model = get_model(settings.model_path)
        win_probability = model.predict_win_probability(payload)
        raw = model.recommend_price(payload, win_probability=win_probability)

        target_margin = (
            payload.target_margin
            if payload.target_margin is not None
            else settings.default_target_margin
        )
        tariff_adjusted_cost = payload.base_cost * (1.0 + payload.tariff_rate)
        # Margin floor: price such that (price - tariff_cost) / price >= target_margin
        # => price >= tariff_cost / (1 - target_margin)
        if target_margin >= 1.0:
            raise HTTPException(status_code=422, detail="target_margin must be < 1.0")
        margin_floor = tariff_adjusted_cost / (1.0 - target_margin)
        # Tariff ceiling: do not exceed list after a modest tariff pass-through buffer.
        tariff_ceiling = payload.list_price * (1.0 + 0.25 * payload.tariff_rate)

        # If the band is empty (rare extreme cost/list cases), widen ceiling to floor.
        if margin_floor > tariff_ceiling:
            logger.warning(
                "infeasible band margin_floor=%.2f tariff_ceiling=%.2f; expanding ceiling",
                margin_floor,
                tariff_ceiling,
            )
            tariff_ceiling = margin_floor

        clamped = clamp_price(raw, margin_floor, tariff_ceiling)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("optimize-price failed")
        raise HTTPException(status_code=500, detail="optimization failed") from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "endpoint=optimize-price latency_ms=%.2f raw=%.2f clamped=%.2f "
        "floor=%.2f ceiling=%.2f win_probability=%.4f",
        latency_ms,
        raw,
        clamped,
        margin_floor,
        tariff_ceiling,
        win_probability,
    )
    return OptimizePriceResponse(
        raw_model_recommendation=raw,
        margin_floor=margin_floor,
        tariff_ceiling=tariff_ceiling,
        clamped_price=clamped,
        win_probability=win_probability,
        latency_ms=latency_ms,
        model_name=model.model_name,
    )
