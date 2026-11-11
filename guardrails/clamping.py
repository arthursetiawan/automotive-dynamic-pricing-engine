"""Price clamping / guardrail utilities."""

from __future__ import annotations


def clamp_price(
    raw_model_recommendation: float,
    margin_floor: float,
    tariff_ceiling: float,
) -> float:
    """Clamp a raw model price into the feasible commercial band.

    Formula:
        clamped_price = max(margin_floor, min(tariff_ceiling, raw_model_recommendation))

    Args:
        raw_model_recommendation: Unconstrained model-suggested selling price.
        margin_floor: Lowest acceptable price that preserves target margin.
        tariff_ceiling: Highest acceptable price after tariff-aware list constraints.

    Returns:
        Feasible clamped selling price.

    Raises:
        ValueError: If inputs are non-finite or the feasible band is empty
            (margin_floor > tariff_ceiling).
    """
    for name, value in (
        ("raw_model_recommendation", raw_model_recommendation),
        ("margin_floor", margin_floor),
        ("tariff_ceiling", tariff_ceiling),
    ):
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            raise ValueError(f"{name} must be a finite number")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    if margin_floor > tariff_ceiling:
        raise ValueError(
            "infeasible band: margin_floor exceeds tariff_ceiling "
            f"({margin_floor} > {tariff_ceiling})"
        )

    return max(margin_floor, min(tariff_ceiling, raw_model_recommendation))


# Alias matching the product requirement naming.
clamped_price = clamp_price
