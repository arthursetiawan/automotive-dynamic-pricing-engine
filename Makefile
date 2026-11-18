.PHONY: train serve test smoke lint typecheck ci docker-build quickstart

PYTHON ?= python

train:
	$(PYTHON) -m models.train --n-samples 20000

serve:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest -q

smoke:
	PYTHONPATH=. $(PYTHON) scripts/smoke_api.py

quickstart: test smoke
	@echo "Ready: run 'make serve' then open http://127.0.0.1:8000/docs"

lint:
	ruff check app models guardrails tests

typecheck:
	mypy app models guardrails

ci: lint typecheck test

docker-build:
	docker build -t automotive-dynamic-pricing-engine .
