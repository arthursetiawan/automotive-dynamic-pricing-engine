"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.routes import health, optimize, predict
from models.inference import clear_model_cache, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    logging.getLogger().setLevel(settings.log_level.upper())
    try:
        load_model(settings.model_path)
        logger.info("Model ready at %s", settings.model_path)
    except FileNotFoundError:
        logger.warning(
            "Model artifact missing at %s; /predict and /optimize-price will return 503 "
            "until `python -m models.train` is run.",
            settings.model_path,
        )
    yield
    clear_model_cache()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = settings

    @application.middleware("http")
    async def latency_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            "method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response

    application.include_router(health.router)
    application.include_router(predict.router)
    application.include_router(optimize.router)

    @application.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "name": settings.app_name,
                "version": __version__,
                "docs": "/docs",
                "health": "/health",
            }
        )

    return application


app = create_app()
