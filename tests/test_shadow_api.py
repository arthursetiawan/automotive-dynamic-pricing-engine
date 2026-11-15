"""API-level shadow logging checks on /optimize-price."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient


def test_optimize_emits_shadow_decision_json(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "part_category": "engine",
        "base_cost": 400.0,
        "list_price": 500.0,
        "requested_discount": 0.35,
        "dealer_tier": "platinum",
        "tariff_rate": 0.15,
        "target_margin": 0.20,
    }
    with caplog.at_level(logging.INFO, logger="app.shadow"):
        response = client.post("/optimize-price", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["clamped_price"] >= body["margin_floor"] - 1e-6

    shadow_msgs = [r.message for r in caplog.records if r.name == "app.shadow"]
    assert shadow_msgs, "expected structured shadow_decision log"
    record = json.loads(shadow_msgs[-1])
    assert record["event"] == "shadow_decision"
    assert "features" in record
    assert "win_prob" in record
    assert "raw_model_recommendation" in record
    assert "margin_floor" in record
    assert "tariff_ceiling" in record
    assert "clamped_price" in record
    assert "clamps_fired" in record
    assert set(record["clamps_fired"]) >= {
        "margin_floor_fired",
        "tariff_ceiling_fired",
        "any_clamp_fired",
    }
