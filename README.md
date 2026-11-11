# Automotive Dynamic Pricing Engine

Production-ready FastAPI service that scores automotive aftermarket part quotes for win probability and recommends a commercially feasible selling price with margin/tariff guardrails.

## Features

- **Strict Pydantic v2 quote validation** (`part_category`, `base_cost`, `list_price`, `requested_discount`, `dealer_tier`, `tariff_rate`)
- **Win-probability model** trained on 20,000 synthetic commercial quotes
  - Prefers **XGBoost**; falls back to **LogisticRegression** if XGBoost is unavailable
  - Bundled in a scikit-learn `Pipeline` + `ColumnTransformer`
  - Serialized to `artifacts/model.joblib` (committed so `/predict` works out of the box)
- **Price guardrails**: `clamped_price = max(margin_floor, min(tariff_ceiling, raw_model_recommendation))`
- **Endpoints**: `/health`, `/predict`, `/optimize-price` with structured latency logging
- **Multi-stage Dockerfile** and GitHub Actions CI (ruff + mypy + pytest)

## Project layout

```
app/                 # FastAPI app, schemas, routes, config
models/              # train.py + inference helpers
guardrails/          # clamping logic
tests/               # pytest suite (schema, clamp, API edge cases)
artifacts/           # model.joblib (committed for out-of-box inference)
.github/workflows/   # CI
Dockerfile           # multi-stage production image
```

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: retrain (overwrites artifacts/model.joblib)
python -m models.train --n-samples 20000

# Serve
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs

### Example requests

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
pytest -q
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
