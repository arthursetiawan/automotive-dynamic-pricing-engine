.PHONY: train serve test evals lint typecheck ci docker-build

train:
	python -m models.train --n-samples 20000

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q tests evals

evals:
	pytest -q evals

lint:
	ruff check app models guardrails tests evals

typecheck:
	mypy app models guardrails

ci: lint typecheck test

docker-build:
	docker build -t automotive-dynamic-pricing-engine .
