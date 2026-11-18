# Model card — win probability (synthetic quotes)

## Model summary

| Field | Value |
|-------|--------|
| **Task** | Binary classification: will this parts quote **win**? |
| **Algorithm** | XGBoost binary classifier when available; otherwise `LogisticRegression` |
| **Preprocessing** | `ColumnTransformer`: one-hot `part_category`, `dealer_tier`; standardize numeric fields |
| **Artifact** | `artifacts/model.joblib` (sklearn `Pipeline` + metadata) |
| **Training script** | `python3 -m models.train` |
| **Inference** | `models/inference.py` → `/predict` and `/optimize-price` |

## Training data

- **Source:** `generate_synthetic_quotes()` in `models/train.py`
- **Samples:** 20,000 rows (default)
- **Split:** 80/20 stratified holdout, `random_state=42`
- **Positive rate (won):** ~47.7%
- **Privacy:** No proprietary or customer data; see [`data/README.md`](../data/README.md)

## Features

`part_category`, `base_cost`, `list_price`, `requested_discount`, `dealer_tier`, `tariff_rate`, `discount_dollars`, `gross_margin_pct`, `tariff_adjusted_cost`.

Derived fields at inference match training (`quote_to_features`).

## Holdout metrics (committed artifact)

Metrics below are from the **held-out 20% test set** bundled in `artifacts/model.joblib` metadata (trained with seed 42, n=20,000).

| Metric | Value |
|--------|-------|
| ROC AUC | 0.664 |
| Accuracy @ 0.5 threshold | 0.619 |
| Train rows | 16,000 |
| Test rows | 4,000 |

Recompute after retraining:

```bash
python3 -c "import joblib; print(joblib.load('artifacts/model.joblib')['metadata']['metrics'])"
```

## Intended use

- **In scope:** Ranking synthetic or similarly structured quotes; pairing win probability with guardrailed price in the demo API.
- **Out of scope:** Production pricing, legal/compliance sign-off, or calibration to a live win-rate baseline.

## Limitations

1. **Synthetic labels** — Win/loss is simulated; real markets have competition, stock, and relationship effects not in the generator.
2. **Moderate AUC** — Holdout ROC ~0.66 reflects overlapping win/loss regions after noise; do not treat scores as calibrated probabilities without isotonic/Platt on real outcomes.
3. **Feature leakage awareness** — `gross_margin_pct` and `discount_dollars` embed quote economics; in production, define features available **at decision time** only.
4. **No drift monitoring** — Artifact is static until retrained; no production retrain pipeline in this repo.
5. **Guardrails are separate** — Margin floor and tariff ceiling are deterministic post-processing; the classifier does not enforce them.
6. **Class balance** — Near 50/50 wins; threshold 0.5 is a demo default, not an optimized commercial cutoff.

## Ethical / compliance notes

- Demonstration portfolio project; no PACCAR or employer metrics.
- Avoid implying deployed fleet savings without measured pilots.

## Related artifacts

- Problem brief: [`docs/problem.md`](../docs/problem.md)
- EDA: [`reports/eda_summary.md`](eda_summary.md) (regenerate via `scripts/eda_synthetic_quotes.py`)