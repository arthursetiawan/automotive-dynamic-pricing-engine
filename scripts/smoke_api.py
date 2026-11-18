#!/usr/bin/env python3
"""Post-install smoke check: model loads and core endpoints respond (no server required)."""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from app.main import app

SAMPLE_QUOTE = {
    "part_category": "brake",
    "base_cost": 80.0,
    "list_price": 160.0,
    "requested_discount": 0.10,
    "dealer_tier": "gold",
    "tariff_rate": 0.05,
}


def main() -> int:
    client = TestClient(app)
    health = client.get("/health")
    if health.status_code != 200 or health.json().get("model_loaded") is not True:
        print("FAIL: /health", health.status_code, health.text, file=sys.stderr)
        return 1

    predict = client.post("/predict", json=SAMPLE_QUOTE)
    if predict.status_code != 200:
        print("FAIL: /predict", predict.status_code, predict.text, file=sys.stderr)
        return 1
    p_body = predict.json()
    if not (0.0 <= p_body["win_probability"] <= 1.0):
        print("FAIL: win_probability out of range", p_body, file=sys.stderr)
        return 1

    optimize = client.post(
        "/optimize-price",
        json={**SAMPLE_QUOTE, "target_margin": 0.12},
    )
    if optimize.status_code != 200:
        print("FAIL: /optimize-price", optimize.status_code, optimize.text, file=sys.stderr)
        return 1
    o_body = optimize.json()
    if o_body["clamped_price"] < o_body["margin_floor"] - 1e-6:
        print("FAIL: clamp below margin floor", o_body, file=sys.stderr)
        return 1

    print("OK: /health, /predict, /optimize-price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())