"""Eval: basic model sanity so win-prob / price recommenders do not silently regress."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import OptimizePriceRequest, QuoteRequest
from guardrails.clamping import clamp_price
from models.inference import ModelBundle, clear_model_cache, load_model


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> ModelBundle:
    from models.train import train_model

    artifact = tmp_path_factory.mktemp("eval_artifacts") / "model.joblib"
    train_model(n_samples=2_000, seed=11, artifact_path=artifact)
    clear_model_cache()
    return load_model(artifact, force_reload=True)


CASES: list[dict] = [
    {
        "part_category": "brake",
        "base_cost": 80.0,
        "list_price": 160.0,
        "requested_discount": 0.10,
        "dealer_tier": "gold",
        "tariff_rate": 0.05,
    },
    {
        "part_category": "engine",
        "base_cost": 400.0,
        "list_price": 520.0,
        "requested_discount": 0.25,
        "dealer_tier": "platinum",
        "tariff_rate": 0.12,
        "target_margin": 0.15,
    },
    {
        "part_category": "filter",
        "base_cost": 15.0,
        "list_price": 30.0,
        "requested_discount": 0.05,
        "dealer_tier": "bronze",
        "tariff_rate": 0.01,
    },
    {
        "part_category": "hvac",
        "base_cost": 120.0,
        "list_price": 200.0,
        "requested_discount": 0.0,
        "dealer_tier": "silver",
        "tariff_rate": 0.0,
    },
]


@pytest.mark.parametrize("case", CASES)
def test_win_probability_in_unit_interval(bundle: ModelBundle, case: dict) -> None:
    quote = QuoteRequest(**{k: v for k, v in case.items() if k != "target_margin"})
    win_prob = bundle.predict_win_probability(quote)
    assert 0.0 <= win_prob <= 1.0


@pytest.mark.parametrize("case", CASES)
def test_raw_recommendation_covers_tariff_cost(bundle: ModelBundle, case: dict) -> None:
    quote = QuoteRequest(**{k: v for k, v in case.items() if k != "target_margin"})
    win_prob = bundle.predict_win_probability(quote)
    raw = bundle.recommend_price(quote, win_probability=win_prob)
    tariff_cost = quote.base_cost * (1.0 + quote.tariff_rate)
    assert raw >= tariff_cost - 1e-6
    assert raw == raw  # not NaN
    assert raw < float("inf")


@pytest.mark.parametrize("case", CASES)
def test_optimize_band_invariants_end_to_end(bundle: ModelBundle, case: dict) -> None:
    payload = OptimizePriceRequest(**case)
    win_prob = bundle.predict_win_probability(payload)
    raw = bundle.recommend_price(payload, win_probability=win_prob)
    target_margin = payload.target_margin if payload.target_margin is not None else 0.12
    tariff_cost = payload.base_cost * (1.0 + payload.tariff_rate)
    floor = tariff_cost / (1.0 - target_margin)
    ceiling = payload.list_price * (1.0 + 0.25 * payload.tariff_rate)
    if floor > ceiling:
        ceiling = floor
    clamped = clamp_price(raw, floor, ceiling)
    assert floor - 1e-9 <= clamped <= ceiling + 1e-9
    assert 0.0 <= win_prob <= 1.0


def test_committed_artifact_loads_when_present() -> None:
    committed = Path("artifacts/model.joblib")
    if not committed.exists():
        pytest.skip("committed model artifact missing")
    clear_model_cache()
    loaded = load_model(committed, force_reload=True)
    assert loaded.model_name
    quote = QuoteRequest(
        part_category="brake",
        base_cost=80.0,
        list_price=160.0,
        requested_discount=0.10,
        dealer_tier="gold",
        tariff_rate=0.05,
    )
    assert 0.0 <= loaded.predict_win_probability(quote) <= 1.0
