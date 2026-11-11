"""API endpoint tests including edge cases."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "X-Latency-Ms" in response.headers


def test_predict_happy_path(client: TestClient, valid_quote: dict) -> None:
    response = client.post("/predict", json=valid_quote)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["recommended_price"] > 0
    assert body["latency_ms"] >= 0
    assert body["model_name"]


def test_optimize_price_applies_floor_on_aggressive_discount(
    client: TestClient,
) -> None:
    # Extreme discount (within platinum cap) that would erode margin.
    payload = {
        "part_category": "engine",
        "base_cost": 400.0,
        "list_price": 500.0,
        "requested_discount": 0.35,
        "dealer_tier": "platinum",
        "tariff_rate": 0.15,
        "target_margin": 0.20,
    }
    response = client.post("/optimize-price", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["clamped_price"] >= body["margin_floor"] - 1e-6
    assert body["clamped_price"] <= body["tariff_ceiling"] + 1e-6
    # Ensure negative-margin attempts get lifted to the floor.
    assert body["clamped_price"] >= body["margin_floor"]


def test_optimize_price_schema_violation(client: TestClient) -> None:
    response = client.post(
        "/optimize-price",
        json={
            "part_category": "brake",
            "base_cost": -5.0,
            "list_price": 100.0,
            "requested_discount": 0.1,
            "dealer_tier": "gold",
            "tariff_rate": 0.05,
        },
    )
    assert response.status_code == 422


def test_predict_rejects_extreme_discount_for_tier(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "part_category": "filter",
            "base_cost": 15.0,
            "list_price": 30.0,
            "requested_discount": 0.5,
            "dealer_tier": "bronze",
            "tariff_rate": 0.01,
        },
    )
    assert response.status_code == 422


def test_predict_rejects_unknown_category(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "part_category": "not-a-part",
            "base_cost": 15.0,
            "list_price": 30.0,
            "requested_discount": 0.05,
            "dealer_tier": "gold",
            "tariff_rate": 0.01,
        },
    )
    assert response.status_code == 422
