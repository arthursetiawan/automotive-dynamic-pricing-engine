"""Training smoke tests."""

from __future__ import annotations

from pathlib import Path

from models.inference import load_model
from models.train import generate_synthetic_quotes, train_model


def test_generate_synthetic_quotes_shape() -> None:
    frame = generate_synthetic_quotes(n_samples=500, seed=1)
    assert len(frame) == 500
    assert set(["part_category", "base_cost", "list_price", "won"]).issubset(frame.columns)
    assert frame["requested_discount"].between(0.0, 0.40).all()


def test_train_writes_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    metadata = train_model(n_samples=1_000, seed=3, artifact_path=artifact)
    assert artifact.exists()
    assert metadata["metrics"]["roc_auc"] > 0.55
    bundle = load_model(artifact, force_reload=True)
    assert bundle.model_name in {"XGBoostClassifier", "LogisticRegression"}
