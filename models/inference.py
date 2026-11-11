"""Model loading and inference helpers used by the API."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import QuoteRequest

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "part_category",
    "base_cost",
    "list_price",
    "requested_discount",
    "dealer_tier",
    "tariff_rate",
    "discount_dollars",
    "gross_margin_pct",
    "tariff_adjusted_cost",
]


class ModelBundle:
    """Thin wrapper around a trained sklearn Pipeline plus metadata."""

    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    def predict_win_probability(self, quote: QuoteRequest) -> float:
        frame = quote_to_features(quote)
        proba = self.pipeline.predict_proba(frame)[0, 1]
        return float(np.clip(proba, 0.0, 1.0))

    def recommend_price(self, quote: QuoteRequest, win_probability: float | None = None) -> float:
        """Derive a raw recommended selling price from quote economics + win prob.

        Uses list price less requested discount, then nudges toward preserving
        a healthy margin when win probability is already high.
        """
        if win_probability is None:
            win_probability = self.predict_win_probability(quote)

        discounted = quote.list_price * (1.0 - quote.requested_discount)
        tariff_cost = quote.base_cost * (1.0 + quote.tariff_rate)
        # High win-prob quotes can hold closer to list; weak quotes lean into discount.
        blend = 0.35 + 0.50 * win_probability
        raw = blend * quote.list_price + (1.0 - blend) * discounted
        # Never recommend below tariff-adjusted cost in the raw suggestion;
        # guardrails may still raise further via margin_floor.
        return float(max(raw, tariff_cost))


_lock = threading.Lock()
_bundle: ModelBundle | None = None
_loaded_path: Path | None = None


def quote_to_features(quote: QuoteRequest) -> pd.DataFrame:
    """Convert a validated quote into the model feature frame."""
    tariff_adjusted_cost = quote.base_cost * (1.0 + quote.tariff_rate)
    discount_dollars = quote.list_price * quote.requested_discount
    selling = quote.list_price - discount_dollars
    gross_margin_pct = 0.0 if selling <= 0 else (selling - tariff_adjusted_cost) / selling

    row = {
        "part_category": quote.part_category.value,
        "base_cost": quote.base_cost,
        "list_price": quote.list_price,
        "requested_discount": quote.requested_discount,
        "dealer_tier": quote.dealer_tier.value,
        "tariff_rate": quote.tariff_rate,
        "discount_dollars": discount_dollars,
        "gross_margin_pct": gross_margin_pct,
        "tariff_adjusted_cost": tariff_adjusted_cost,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def load_model(model_path: Path | str, *, force_reload: bool = False) -> ModelBundle:
    """Load (or return cached) model bundle from disk."""
    global _bundle, _loaded_path
    path = Path(model_path)
    with _lock:
        if _bundle is not None and _loaded_path == path and not force_reload:
            return _bundle
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. "
                "Run `python -m models.train` to generate artifacts/model.joblib."
            )
        payload = joblib.load(path)
        if isinstance(payload, dict) and "pipeline" in payload:
            bundle = ModelBundle(pipeline=payload["pipeline"], metadata=payload.get("metadata", {}))
        else:
            bundle = ModelBundle(pipeline=payload, metadata={"model_name": type(payload).__name__})
        _bundle = bundle
        _loaded_path = path
        logger.info("Loaded model from %s (%s)", path, bundle.model_name)
        return bundle


def get_model(model_path: Path | str) -> ModelBundle:
    """Return the cached model, loading it if necessary."""
    return load_model(model_path)


def clear_model_cache() -> None:
    """Clear the in-process model cache (used in tests)."""
    global _bundle, _loaded_path
    with _lock:
        _bundle = None
        _loaded_path = None
