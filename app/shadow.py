"""Structured shadow-decision logging for price optimization."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.schemas import OptimizePriceRequest

logger = logging.getLogger("app.shadow")


def clamp_flags(
    raw_model_recommendation: float,
    margin_floor: float,
    tariff_ceiling: float,
    clamped_price: float,
) -> dict[str, bool]:
    """Derive whether margin-floor / tariff-ceiling clamps fired."""
    floor_fired = raw_model_recommendation < margin_floor - 1e-9
    ceiling_fired = raw_model_recommendation > tariff_ceiling + 1e-9
    # Consistency check against the clamped result (floating tolerance).
    if abs(clamped_price - max(margin_floor, min(tariff_ceiling, raw_model_recommendation))) > 1e-6:
        floor_fired = clamped_price > raw_model_recommendation + 1e-9
        ceiling_fired = clamped_price < raw_model_recommendation - 1e-9
    return {
        "margin_floor_fired": floor_fired,
        "tariff_ceiling_fired": ceiling_fired,
        "any_clamp_fired": floor_fired or ceiling_fired,
    }


def request_features(payload: OptimizePriceRequest) -> dict[str, Any]:
    """Serialize request features used for shadow decision audit."""
    features: dict[str, Any] = {
        "part_category": payload.part_category.value,
        "base_cost": payload.base_cost,
        "list_price": payload.list_price,
        "requested_discount": payload.requested_discount,
        "dealer_tier": payload.dealer_tier.value,
        "tariff_rate": payload.tariff_rate,
    }
    if payload.target_margin is not None:
        features["target_margin"] = payload.target_margin
    return features


def log_shadow_decision(
    *,
    payload: OptimizePriceRequest,
    win_prob: float,
    raw_model_recommendation: float,
    margin_floor: float,
    tariff_ceiling: float,
    clamped_price: float,
    model_name: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Emit one structured JSON shadow-decision log line and return the payload."""
    flags = clamp_flags(
        raw_model_recommendation,
        margin_floor,
        tariff_ceiling,
        clamped_price,
    )
    record: dict[str, Any] = {
        "event": "shadow_decision",
        "features": request_features(payload),
        "win_prob": round(float(win_prob), 6),
        "raw_model_recommendation": round(float(raw_model_recommendation), 4),
        "margin_floor": round(float(margin_floor), 4),
        "tariff_ceiling": round(float(tariff_ceiling), 4),
        "clamped_price": round(float(clamped_price), 4),
        "clamps_fired": flags,
        "model_name": model_name,
        "latency_ms": round(float(latency_ms), 2),
    }
    logger.info("%s", json.dumps(record, separators=(",", ":")))
    return record
