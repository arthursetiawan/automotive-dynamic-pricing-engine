# syntax=docker/dockerfile:1.7

# ---- builder: install deps + optionally train model ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY app ./app
COPY models ./models
COPY guardrails ./guardrails
COPY artifacts ./artifacts
COPY pyproject.toml README.md ./

# Prefer the committed artifact; train only if missing (keeps builds deterministic/fast).
RUN if [ ! -f artifacts/model.joblib ]; then python -m models.train --n-samples 20000; fi

# ---- runtime: slim image without build tools ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ADPE_HOST=0.0.0.0 \
    ADPE_PORT=8000 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/app ./app
COPY --from=builder /build/models ./models
COPY --from=builder /build/guardrails ./guardrails
COPY --from=builder /build/artifacts ./artifacts
COPY --from=builder /build/pyproject.toml /build/README.md ./

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
