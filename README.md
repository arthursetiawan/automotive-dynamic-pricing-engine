# Automotive Dynamic Pricing Engine

## Charter

| | |
|---|---|
| **Business idea** | Teachable **parts-style dynamic pricing**: ML win-probability on a quote plus deterministic **margin floor** and **tariff ceiling** guardrails—patterns familiar from aftermarket pricing, expressed on open synthetic data. |
| **North star** | `POST /predict` and `POST /optimize-price` on synthetic quotes, bundled model artifact, multi-stage **Docker** image, green **GitHub Actions** CI. |
| **Live demo (portfolio)** | https://arthursetiawan.github.io/demos/automotive-pricing/ — UI mock; **this repo** is the runnable API and training code. |
| **Non-goals** | No PACCAR (or other employer) proprietary data, internal architectures, or production metrics. **Resume-honest stack only:** Python 3.11+, FastAPI, Pydantic v2, scikit-learn / XGBoost, pytest, ruff, mypy, Docker. |

## Business outcome

Pricing analysts need to respond to a quote with both **how likely it is to win** and **a price that still clears margin and tariff floors**. This service scores each synthetic line-item quote with a trained win model and returns a guardrailed optimized price in one API round-trip—so demos show faster, constraint-aware decisions instead of spreadsheet guesswork. Holdout quality and segment behavior are documented in [`reports/model_card.md`](reports/model_card.md) and [`reports/eda_summary.md`](reports/eda_summary.md); the customer problem and KPIs are in [`docs/problem.md`](docs/problem.md).

---

## Run locally in under 5 minutes

Prerequisites: **Python 3.11+** and `git`.

```bash
git clone https://github.com/arthursetiawan/automotive-dynamic-pricing-engine.git
cd automotive-dynamic-pricing-engine
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make quickstart    # pytest + API smoke (uses committed artifacts/model.joblib)
make serve         # http://127.0.0.1:8000/docs
```

One-liner check without starting the server:

```bash
make smoke
```

The committed `artifacts/model.joblib` avoids a training step on first clone. To regenerate the model (optional, ~1–2 min):

```bash
python -m models.train --n-samples 20000
```

---

Production-ready FastAPI service that scores automotive aftermarket part quotes for win probability and recommends a commercially feasible selling price with margin/tariff guardrails.

## Features

- **Strict Pydantic v2 quote validation** (`part_category`, `base_cost`, `list_price`, `requested_discount`, `dealer_tier`, `tariff_rate`)
- **Win-probability model** trained on 20,000 synthetic commercial quotes
  - Prefers **XGBoost**; falls back to **LogisticRegression** if XGBoost is unavailable
  - Bundled in a scikit-learn `Pipeline` + `ColumnTransformer`
  - Serialized to `artifacts/model.joblib` (committed so `/predict` works out of the box)
- **Price guardrails**: `clamped_price = max(margin_floor, min(tariff_ceiling, raw_model_recommendation))`
- **Endpoints**: `/health`, `/predict`, `/optimize-price` with structured latency logging
- **Multi-stage Dockerfile** and GitHub Actions CI (ruff + mypy + pytest + API smoke)

## Synthetic data (no proprietary inputs)

All training labels come from **`models/train.py`**: `generate_synthetic_quotes()` samples part categories, dealer tiers, list/margin structure, and tariff rates, then assigns a win/loss outcome from a transparent logistic-style rule (discount depth vs tier caps, margin, noise). Nothing is downloaded from production systems. See `python -m models.train --help` for CLI options.

## Project layout

```
app/                 # FastAPI app, schemas, routes, config
models/              # train.py + inference helpers
guardrails/          # clamping logic
scripts/             # smoke_api.py, eda_synthetic_quotes.py
docs/                # problem brief (customer, KPI, non-goals)
data/                # README: schema + reproduce synthetic quotes (no committed rows)
reports/             # eda_summary.md, model_card.md
tests/               # pytest suite (schema, clamp, API edge cases)
artifacts/           # model.joblib (committed for out-of-box inference)
.github/workflows/   # CI
Dockerfile           # multi-stage production image
```

## Example requests

With the server running (`make serve`):

```bash
curl -s http://127.0.0.1:8000/health | jq

curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "part_category": "brake",
    "base_cost": 80.0,
    "list_price": 160.0,
    "requested_discount": 0.10,
    "dealer_tier": "gold",
    "tariff_rate": 0.05
  }' | jq

curl -s http://127.0.0.1:8000/optimize-price \
  -H 'Content-Type: application/json' \
  -d '{
    "part_category": "engine",
    "base_cost": 400.0,
    "list_price": 520.0,
    "requested_discount": 0.25,
    "dealer_tier": "platinum",
    "tariff_rate": 0.12,
    "target_margin": 0.15
  }' | jq
```

## Train

```bash
python -m models.train --n-samples 20000 --artifact artifacts/model.joblib
```

Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `ADPE_MODEL_PATH` / `MODEL_PATH` | `artifacts/model.joblib` | Model artifact path |
| `ADPE_DEFAULT_TARGET_MARGIN` | `0.12` | Default margin floor fraction |
| `ADPE_LOG_LEVEL` | `INFO` | Logging level |

## Test

```bash
make test      # full pytest suite
make smoke     # fast path: health + predict + optimize-price
make ci        # lint + typecheck + test (local mirror of CI minus Docker)
```

CI trains a **small fixture model only if** `artifacts/model.joblib` is missing. The committed artifact keeps CI and local `/predict` fast.

## Docker

```bash
docker build -t automotive-dynamic-pricing-engine .
docker run --rm -p 8000:8000 automotive-dynamic-pricing-engine
```

The image reuses the committed model when present; otherwise the builder stage trains one.

## API summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + model load status |
| `POST` | `/predict` | Win probability + raw recommended price |
| `POST` | `/optimize-price` | Clamped price with margin floor & tariff ceiling |

## License

MIT