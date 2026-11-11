"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.inference import clear_model_cache
from models.train import train_model


@pytest.fixture(scope="session")
def model_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small fixture model once per test session for speed."""
    artifact = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    train_model(n_samples=2_000, seed=7, artifact_path=artifact)
    return artifact


@pytest.fixture()
def client(model_artifact: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    clear_model_cache()
    monkeypatch.setenv("ADPE_MODEL_PATH", str(model_artifact))
    # Reset settings cache and re-import app with patched path.
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MODEL_PATH", str(model_artifact))

    from app.main import create_app

    application = create_app()
    application.state.settings.model_path = model_artifact
    with TestClient(application) as test_client:
        yield test_client
    clear_model_cache()
    get_settings.cache_clear()


@pytest.fixture()
def valid_quote() -> dict:
    return {
        "part_category": "brake",
        "base_cost": 80.0,
        "list_price": 160.0,
        "requested_discount": 0.10,
        "dealer_tier": "gold",
        "tariff_rate": 0.05,
    }
