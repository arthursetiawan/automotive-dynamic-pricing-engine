# Problem brief — parts-style dynamic pricing

## Customer

Aftermarket **parts pricing analysts** and commercial managers at dealers or distributors. They field line-item quotes (category, list price, discount request, dealer tier) and must respond quickly without eroding margin or blowing past tariff-adjusted cost.

## Pain

- **Too high** → the quote loses to a competitor.
- **Too low** → the deal wins but gross margin falls below the floor.
- Analysts lack a single scored view: **win likelihood** plus a **guardrailed** price recommendation they can defend.

## Success metric (KPI)

| KPI | Definition | Target direction |
|-----|------------|------------------|
| **Quote win rate** | Share of quotes marked won in a holdout or pilot cohort | Increase vs. baseline manual pricing |
| **Margin compliance** | Share of accepted quotes at or above the configured margin floor | ≥ 95% |
| **Decision latency** | Time from quote request to priced response via API | Decrease (seconds, not spreadsheet hours) |

Portfolio demo success: recruiters and hiring managers can run `POST /predict` and `POST /optimize-price` on **synthetic** quotes and see win probability + clamped price in one response.

## Scope (this repo)

- Trainable win-probability model on **synthetic** commercial quotes (`models/train.py`).
- Deterministic guardrails: margin floor and tariff ceiling (`guardrails/clamping.py`).
- FastAPI service with CI, Docker, and smoke tests.

## Non-goals

- No PACCAR or other employer proprietary data, internal architectures, or production KPIs.
- No live ERP, catalog, or competitor price feeds.
- No claim of fleet-wide or dollarized savings from a real deployment.
- No black-box auto-pricing without human-readable guardrails and schema validation.

## Attention hook (customer moment)

> “What’s the win probability on this quote—and what’s the best price that still respects our margin floor?”

One API call returns both signals so pricing can move faster with explicit commercial constraints.