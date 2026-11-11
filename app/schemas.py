"""Pydantic v2 request/response schemas with strict validation."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    field_validator,
    model_validator,
)


class PartCategory(str, Enum):
    """Supported automotive aftermarket part categories."""

    BRAKE = "brake"
    ENGINE = "engine"
    ELECTRICAL = "electrical"
    BODY = "body"
    SUSPENSION = "suspension"
    TRANSMISSION = "transmission"
    FILTER = "filter"
    HVAC = "hvac"


class DealerTier(str, Enum):
    """Dealer partnership tiers used for discount eligibility."""

    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


NonNegativeFloat = Annotated[StrictFloat, Field(ge=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PositiveFloat = Annotated[StrictFloat, Field(gt=0.0, allow_inf_nan=False)]


class QuoteRequest(BaseModel):
    """Inbound automotive part quote for win-probability scoring."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    part_category: PartCategory
    base_cost: NonNegativeFloat = Field(description="Unit landed cost before tariff.")
    list_price: PositiveFloat = Field(description="Published list / MSRP.")
    requested_discount: UnitInterval = Field(
        description="Requested off-list discount as a fraction in [0, 1]."
    )
    dealer_tier: DealerTier
    tariff_rate: UnitInterval = Field(
        description="Applicable tariff rate as a fraction in [0, 1]."
    )

    @field_validator("base_cost", "list_price", "requested_discount", "tariff_rate", mode="before")
    @classmethod
    def _reject_bools(cls, value: object) -> object:
        # Bools are ints in Python; reject explicitly for quote numerics.
        if isinstance(value, bool):
            raise ValueError("boolean values are not allowed for numeric fields")
        return value

    @model_validator(mode="after")
    def _validate_price_relationships(self) -> QuoteRequest:
        if self.list_price < self.base_cost:
            raise ValueError("list_price must be greater than or equal to base_cost")
        max_discount = _max_discount_for_tier(self.dealer_tier)
        if self.requested_discount > max_discount:
            raise ValueError(
                f"requested_discount {self.requested_discount:.2f} exceeds "
                f"max {max_discount:.2f} for dealer_tier={self.dealer_tier.value}"
            )
        return self


class PredictResponse(BaseModel):
    """Win-probability prediction payload."""

    model_config = ConfigDict(extra="forbid")

    win_probability: float = Field(ge=0.0, le=1.0)
    recommended_price: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    model_name: str


class OptimizePriceRequest(QuoteRequest):
    """Quote plus optional margin floor override for price optimization."""

    target_margin: UnitInterval | None = Field(
        default=None,
        description="Optional minimum margin as a fraction of recommended price.",
    )


class OptimizePriceResponse(BaseModel):
    """Clamped price optimization result."""

    model_config = ConfigDict(extra="forbid")

    raw_model_recommendation: float = Field(ge=0.0)
    margin_floor: float = Field(ge=0.0)
    tariff_ceiling: float = Field(ge=0.0)
    clamped_price: float = Field(ge=0.0)
    win_probability: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_name: str


class HealthResponse(BaseModel):
    """Service health payload."""

    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_path: str
    version: str


def _max_discount_for_tier(tier: DealerTier) -> float:
    limits = {
        DealerTier.PLATINUM: 0.40,
        DealerTier.GOLD: 0.30,
        DealerTier.SILVER: 0.20,
        DealerTier.BRONZE: 0.10,
    }
    return limits[tier]
