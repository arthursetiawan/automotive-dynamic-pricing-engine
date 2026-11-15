"""Eval/unit coverage for structured shadow decision logging."""

from __future__ import annotations

import json
import logging

import pytest

from app.schemas import OptimizePriceRequest
from app.shadow import log_shadow_decision, request_features


def test_request_features_include_core_fields() -> None:
    payload = OptimizePriceRequest(
        part_category="brake",
        base_cost=80.0,
        list_price=160.0,
        requested_discount=0.10,
        dealer_tier="gold",
        tariff_rate=0.05,
        target_margin=0.18,
    )
    features = request_features(payload)
    assert features["part_category"] == "brake"
    assert features["dealer_tier"] == "gold"
    assert features["target_margin"] == 0.18


def test_shadow_log_emits_structured_json(caplog: pytest.LogCaptureFixture) -> None:
    payload = OptimizePriceRequest(
        part_category="engine",
        base_cost=400.0,
        list_price=500.0,
        requested_discount=0.35,
        dealer_tier="platinum",
        tariff_rate=0.15,
        target_margin=0.20,
    )
    with caplog.at_level(logging.INFO, logger="app.shadow"):
        record = log_shadow_decision(
            payload=payload,
            win_prob=0.42,
            raw_model_recommendation=70.0,
            margin_floor=95.0,
            tariff_ceiling=150.0,
            clamped_price=95.0,
            model_name="fixture",
            latency_ms=1.25,
        )

    assert record["event"] == "shadow_decision"
    assert record["win_prob"] == 0.42
    assert record["raw_model_recommendation"] == 70.0
    assert record["margin_floor"] == 95.0
    assert record["tariff_ceiling"] == 150.0
    assert record["clamped_price"] == 95.0
    assert record["clamps_fired"]["margin_floor_fired"] is True
    assert record["clamps_fired"]["tariff_ceiling_fired"] is False
    assert record["clamps_fired"]["any_clamp_fired"] is True
    assert "features" in record

    json_lines = [r.message for r in caplog.records if r.name == "app.shadow"]
    assert json_lines
    parsed = json.loads(json_lines[-1])
    assert parsed["event"] == "shadow_decision"
    assert parsed["clamps_fired"]["any_clamp_fired"] is True
