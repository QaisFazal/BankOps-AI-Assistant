"""Global exception handlers for the API."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.security.rate_limiter import RateLimitExceeded


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach app-wide handlers for unexpected errors."""

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        """Return a graceful 429 response with an activity log entry."""

        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "Rate limit exceeded",
            extra={
                "component": "api",
                "operation": "rate_limit",
                "error_type": type(exc).__name__,
                "fallback": "http_429",
                "request_id": request_id,
                "user_id": exc.user_id,
                "path": request.url.path,
                "method": request.method,
            },
        )
        activity_log = [
            {
                "node": "rate_limiter",
                "status": "blocked",
                "message": (
                    f"Rate limit exceeded for user {exc.user_id}. "
                    f"Retry after {exc.retry_after_seconds:.2f} second(s)."
                ),
            }
        ]
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(int(exc.retry_after_seconds) + 1)},
            content={
                "detail": "Rate limit exceeded. Please try again shortly.",
                "request_id": request_id,
                "activity_log": activity_log,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a safe error response instead of exposing internals."""

        logger.exception(
            "Unhandled API exception",
            extra={
                "component": "api",
                "operation": "request_handler",
                "error_type": type(exc).__name__,
                "fallback": "http_500_safe_response",
                "method": request.method,
                "path": request.url.path,
                "request_id": getattr(request.state, "request_id", None),
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )
