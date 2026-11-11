"""Schema validation edge-case tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import QuoteRequest


def test_valid_quote_parses() -> None:
    quote = QuoteRequest(
        part_category="brake",
        base_cost=80.0,
        list_price=160.0,
        requested_discount=0.1,
        dealer_tier="gold",
        tariff_rate=0.05,
    )
    assert quote.part_category.value == "brake"


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_cost": -1.0},
        {"list_price": -10.0},
        {"list_price": 0.0},
        {"requested_discount": -0.01},
        {"requested_discount": 1.01},
        {"tariff_rate": -0.1},
        {"tariff_rate": 1.5},
        {"part_category": "tires"},
        {"dealer_tier": "diamond"},
        {"base_cost": True},
        {"list_price": False},
    ],
)
def test_schema_violations(overrides: dict) -> None:
    payload = {
        "part_category": "brake",
        "base_cost": 80.0,
        "list_price": 160.0,
        "requested_discount": 0.1,
        "dealer_tier": "gold",
        "tariff_rate": 0.05,
    }
    payload.update(overrides)
    with pytest.raises(ValidationError):
        QuoteRequest(**payload)


def test_list_price_below_base_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        QuoteRequest(
            part_category="engine",
            base_cost=500.0,
            list_price=400.0,
            requested_discount=0.05,
            dealer_tier="silver",
            tariff_rate=0.0,
        )


def test_extreme_discount_above_tier_cap_rejected() -> None:
    with pytest.raises(ValidationError):
        QuoteRequest(
            part_category="filter",
            base_cost=20.0,
            list_price=40.0,
            requested_discount=0.25,  # bronze max is 0.10
            dealer_tier="bronze",
            tariff_rate=0.02,
        )


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        QuoteRequest(
            part_category="brake",
            base_cost=80.0,
            list_price=160.0,
            requested_discount=0.1,
            dealer_tier="gold",
            tariff_rate=0.05,
            sneaky="nope",  # type: ignore[call-arg]
        )
