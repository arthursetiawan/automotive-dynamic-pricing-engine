# Synthetic quote data

All training and EDA in this repository use **generated** data only. Nothing is copied from production systems, employer IP, or customer uploads.

## Source

Quotes are created in code by `generate_synthetic_quotes()` in [`models/train.py`](../models/train.py). The generator:

1. Samples **part category** (8 aftermarket families), **dealer tier** (platinum → bronze), costs, list prices, tariff rates, and requested discounts (capped per tier).
2. Computes derived fields: `discount_dollars`, `tariff_adjusted_cost`, `gross_margin_pct`.
3. Assigns **`won`** (0/1) from a transparent logistic-style rule: higher margin and lighter discounts increase win chance; deep discounts and weak tiers decrease it, plus Gaussian noise.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `part_category` | string | One of: brake, engine, electrical, body, suspension, transmission, filter, hvac |
| `base_cost` | float | Unit cost before tariff |
| `list_price` | float | Published list (markup over cost) |
| `requested_discount` | float | Fraction off list (0–tier cap) |
| `dealer_tier` | string | platinum, gold, silver, bronze |
| `tariff_rate` | float | Fraction added to cost (0–0.35 sampled) |
| `discount_dollars` | float | `list_price * requested_discount` |
| `tariff_adjusted_cost` | float | `base_cost * (1 + tariff_rate)` |
| `gross_margin_pct` | float | Margin on selling price after discount vs. tariff-adjusted cost |
| `won` | int | 1 = win, 0 = loss (synthetic label) |

**Model features** (API + training): all columns except `won` map to `FEATURE_COLUMNS` in [`models/inference.py`](../models/inference.py).

## Reproduce

### Generate rows in Python

```bash
python3 -c "
from models.train import generate_synthetic_quotes
df = generate_synthetic_quotes(n_samples=5000, seed=42)
print(df.head())
print('win rate', df['won'].mean())
"
```

### Train model + write artifact

```bash
python3 -m models.train --n-samples 20000 --seed 42 --artifact artifacts/model.joblib
```

### EDA report (win rate by segment, margin bands)

```bash
python3 scripts/eda_synthetic_quotes.py
# writes reports/eda_summary.md
```

## Local exports (optional)

This directory is **gitignored** except this README. You may write local CSV/JSONL exports here for notebooks, e.g.:

```bash
python3 -c "
from models.train import generate_synthetic_quotes
generate_synthetic_quotes(10000, 42).to_csv('data/synthetic_quotes.csv', index=False)
"
```

Do not commit real customer or employer data.

## Privacy

- Fixed random seed (`42` default) for reproducible demos.
- Distributions are illustrative, not calibrated to any OEM or fleet.