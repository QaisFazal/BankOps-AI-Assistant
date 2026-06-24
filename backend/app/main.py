"""FastAPI entrypoint for the AI Lead Assistant backend.

This file is intentionally small for the first assignment milestone. The
feature-specific code lives in subpackages so the application can grow without
turning this file into a grab bag.
"""

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Starter backend for an enterprise AI assistant.",
)

app.include_router(router, prefix="/api")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return a tiny health response for local smoke tests and containers."""

    return {"status": "ok", "environment": settings.environment}
