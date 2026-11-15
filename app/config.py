"""Application configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"

DEFAULT_CORS_ORIGINS = [
    "https://arthursetiawan.github.io",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="ADPE_", env_file=".env", extra="ignore")

    app_name: str = "Automotive Dynamic Pricing Engine"
    model_path: Path = Field(default=DEFAULT_MODEL_PATH)
    default_target_margin: float = Field(default=0.12, ge=0.0, le=1.0)
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    shadow_logging: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        if value is None or value == "":
            return list(DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return [part for part in parts if part]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    # Allow MODEL_PATH override without prefix for Docker convenience.
    path_override = os.getenv("MODEL_PATH")
    settings = Settings()
    if path_override:
        settings.model_path = Path(path_override)
    return settings
