"""FastAPI entrypoint for the AI Lead Assistant backend."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Request

from app.api.exception_handlers import register_exception_handlers
from app.api.routes import router
from app.config import get_settings
from app.observability.logging import configure_logging
from app.observability.tracing import configure_tracing


configure_logging()
settings = get_settings()
configure_tracing(settings)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Starter backend for an enterprise AI assistant.",
)

register_exception_handlers(app)
app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each request as structured JSON with a request id."""

    request_id = str(uuid4())
    request.state.request_id = request_id
    start_time = perf_counter()

    response = await call_next(request)
    duration_ms = round((perf_counter() - start_time) * 1000, 2)

    logger.info(
        "HTTP request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return a tiny health response for local smoke tests and containers."""

    return {"status": "ok", "environment": settings.environment}
