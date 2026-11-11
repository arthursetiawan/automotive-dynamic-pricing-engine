"""Guardrail clamping tests."""

from __future__ import annotations

import pytest

from guardrails.clamping import clamp_price


def test_clamp_between_floor_and_ceiling() -> None:
    assert clamp_price(100.0, 90.0, 120.0) == 100.0


def test_clamp_raises_to_margin_floor() -> None:
    # Negative-margin attempt: raw recommendation below floor.
    assert clamp_price(70.0, 95.0, 140.0) == 95.0


def test_clamp_caps_at_tariff_ceiling() -> None:
    assert clamp_price(200.0, 95.0, 150.0) == 150.0


def test_infeasible_band_raises() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        clamp_price(100.0, 160.0, 150.0)


@pytest.mark.parametrize(
    "raw, floor, ceiling",
    [
        (float("nan"), 1.0, 2.0),
        (1.0, float("inf"), 2.0),
        (-1.0, 0.0, 2.0),
        (1.0, -0.1, 2.0),
    ],
)
def test_invalid_inputs_raise(raw: float, floor: float, ceiling: float) -> None:
    with pytest.raises(ValueError):
        clamp_price(raw, floor, ceiling)
