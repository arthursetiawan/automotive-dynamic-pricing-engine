"""Synthetic dataset generation and win-probability model training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.inference import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT_DIR / "artifacts" / "model.joblib"

PART_CATEGORIES = [
    "brake",
    "engine",
    "electrical",
    "body",
    "suspension",
    "transmission",
    "filter",
    "hvac",
]
DEALER_TIERS = ["platinum", "gold", "silver", "bronze"]
TIER_DISCOUNT_CAP = {"platinum": 0.40, "gold": 0.30, "silver": 0.20, "bronze": 0.10}
CATEGORY_COST_MU = {
    "brake": 85,
    "engine": 420,
    "electrical": 140,
    "body": 260,
    "suspension": 190,
    "transmission": 510,
    "filter": 28,
    "hvac": 220,
}


def _try_xgboost_classifier() -> Any | None:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=42,
        )
    except Exception as exc:  # noqa: BLE001 - documented fallback path
        logger.warning("XGBoost unavailable (%s); falling back to LogisticRegression.", exc)
        return None


def generate_synthetic_quotes(n_samples: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a clean synthetic commercial parts-quote dataset."""
    rng = np.random.default_rng(seed)
    categories = rng.choice(PART_CATEGORIES, size=n_samples)
    tiers = rng.choice(DEALER_TIERS, size=n_samples, p=[0.15, 0.30, 0.35, 0.20])

    base_costs = np.empty(n_samples, dtype=float)
    for i, cat in enumerate(categories):
        mu = CATEGORY_COST_MU[str(cat)]
        base_costs[i] = float(np.clip(rng.lognormal(np.log(mu), 0.35), mu * 0.35, mu * 4.0))

    markup = rng.uniform(1.25, 2.40, size=n_samples)
    list_prices = base_costs * markup
    tariff_rates = np.clip(rng.beta(2.0, 12.0, size=n_samples), 0.0, 0.35)

    requested = np.empty(n_samples, dtype=float)
    for i, tier in enumerate(tiers):
        cap = TIER_DISCOUNT_CAP[str(tier)]
        requested[i] = float(rng.uniform(0.0, cap))

    tariff_adjusted_cost = base_costs * (1.0 + tariff_rates)
    discount_dollars = list_prices * requested
    selling = list_prices - discount_dollars
    gross_margin_pct = (selling - tariff_adjusted_cost) / np.maximum(selling, 1e-6)

    # Latent win propensity: healthier margins + lighter discounts + better tiers win more.
    tier_bonus = np.array(
        [{"platinum": 0.18, "gold": 0.10, "silver": 0.03, "bronze": -0.05}[str(t)] for t in tiers]
    )
    logit = (
        -0.35
        + 2.8 * gross_margin_pct
        - 3.2 * requested
        + tier_bonus
        - 1.1 * tariff_rates
        + rng.normal(0.0, 0.55, size=n_samples)
    )
    win_prob = 1.0 / (1.0 + np.exp(-logit))
    won = rng.binomial(1, win_prob)

    frame = pd.DataFrame(
        {
            "part_category": categories,
            "base_cost": base_costs,
            "list_price": list_prices,
            "requested_discount": requested,
            "dealer_tier": tiers,
            "tariff_rate": tariff_rates,
            "discount_dollars": discount_dollars,
            "gross_margin_pct": gross_margin_pct,
            "tariff_adjusted_cost": tariff_adjusted_cost,
            "won": won,
        }
    )
    return frame


def build_pipeline(classifier: Any) -> Pipeline:
    """Build a sklearn Pipeline with ColumnTransformer preprocessing."""
    categorical = ["part_category", "dealer_tier"]
    numeric = [
        "base_cost",
        "list_price",
        "requested_discount",
        "tariff_rate",
        "discount_dollars",
        "gross_margin_pct",
        "tariff_adjusted_cost",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", classifier),
        ]
    )


def train_model(
    n_samples: int = 20_000,
    seed: int = 42,
    artifact_path: Path | str = DEFAULT_ARTIFACT,
) -> dict[str, Any]:
    """Train win-probability classifier and serialize to joblib."""
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_quotes(n_samples=n_samples, seed=seed)
    x = data[FEATURE_COLUMNS]
    y = data["won"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )

    xgb = _try_xgboost_classifier()
    if xgb is not None:
        classifier = xgb
        model_name = "XGBoostClassifier"
    else:
        classifier = LogisticRegression(max_iter=2000, random_state=seed)
        model_name = "LogisticRegression"

    pipeline = build_pipeline(classifier)
    pipeline.fit(x_train, y_train)

    proba = pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "positive_rate": float(y.mean()),
    }

    payload = {
        "pipeline": pipeline,
        "metadata": {
            "model_name": model_name,
            "feature_columns": FEATURE_COLUMNS,
            "n_samples": n_samples,
            "seed": seed,
            "metrics": metrics,
        },
    }
    joblib.dump(payload, artifact)
    logger.info(
        "Wrote %s (%s) roc_auc=%.4f accuracy=%.4f",
        artifact,
        model_name,
        metrics["roc_auc"],
        metrics["accuracy"],
    )
    return payload["metadata"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train automotive win-probability model")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    metadata = train_model(n_samples=args.n_samples, seed=args.seed, artifact_path=args.artifact)
    print(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
