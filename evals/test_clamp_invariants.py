"""Eval: clamp invariants must hold for synthetic commercial quotes."""

from __future__ import annotations

import math

import pytest

from app.schemas import DealerTier, OptimizePriceRequest, PartCategory
from app.shadow import clamp_flags
from guardrails.clamping import clamp_price


def _band(
    base_cost: float,
    list_price: float,
    tariff_rate: float,
    target_margin: float,
) -> tuple[float, float]:
    tariff_cost = base_cost * (1.0 + tariff_rate)
    margin_floor = tariff_cost / (1.0 - target_margin)
    tariff_ceiling = list_price * (1.0 + 0.25 * tariff_rate)
    if margin_floor > tariff_ceiling:
        tariff_ceiling = margin_floor
    return margin_floor, tariff_ceiling


@pytest.mark.parametrize(
    "raw,base_cost,list_price,tariff_rate,target_margin",
    [
        (100.0, 80.0, 160.0, 0.05, 0.12),
        (50.0, 80.0, 160.0, 0.05, 0.20),  # below floor
        (300.0, 80.0, 160.0, 0.05, 0.12),  # above ceiling
        (120.0, 400.0, 500.0, 0.15, 0.20),  # aggressive discount band
        (520.0, 400.0, 520.0, 0.12, 0.15),
        (10.0, 15.0, 30.0, 0.01, 0.10),
    ],
)
def test_clamped_price_within_band(
    raw: float,
    base_cost: float,
    list_price: float,
    tariff_rate: float,
    target_margin: float,
) -> None:
    floor, ceiling = _band(base_cost, list_price, tariff_rate, target_margin)
    clamped = clamp_price(raw, floor, ceiling)
    assert floor - 1e-9 <= clamped <= ceiling + 1e-9
    expected = max(floor, min(ceiling, raw))
    assert math.isclose(clamped, expected, rel_tol=0, abs_tol=1e-9)


@pytest.mark.parametrize(
    "raw,floor,ceiling,expect_floor,expect_ceiling",
    [
        (100.0, 90.0, 120.0, False, False),
        (70.0, 95.0, 140.0, True, False),
        (200.0, 95.0, 150.0, False, True),
        (95.0, 95.0, 150.0, False, False),
        (150.0, 95.0, 150.0, False, False),
    ],
)
def test_clamp_flags_match_raw_vs_band(
    raw: float,
    floor: float,
    ceiling: float,
    expect_floor: bool,
    expect_ceiling: bool,
) -> None:
    clamped = clamp_price(raw, floor, ceiling)
    flags = clamp_flags(raw, floor, ceiling, clamped)
    assert flags["margin_floor_fired"] is expect_floor
    assert flags["tariff_ceiling_fired"] is expect_ceiling
    assert flags["any_clamp_fired"] is (expect_floor or expect_ceiling)


def test_optimize_request_features_round_trip() -> None:
    payload = OptimizePriceRequest(
        part_category=PartCategory.BRAKE,
        base_cost=80.0,
        list_price=160.0,
        requested_discount=0.10,
        dealer_tier=DealerTier.GOLD,
        tariff_rate=0.05,
        target_margin=0.15,
    )
    floor, ceiling = _band(
        payload.base_cost,
        payload.list_price,
        payload.tariff_rate,
        payload.target_margin or 0.12,
    )
    raw = payload.list_price * (1.0 - payload.requested_discount)
    clamped = clamp_price(raw, floor, ceiling)
    assert clamped >= floor - 1e-9
    assert clamped <= ceiling + 1e-9
