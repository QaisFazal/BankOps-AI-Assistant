"""Global exception handlers for the API."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach app-wide handlers for unexpected errors."""

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a safe error response instead of exposing internals."""

        logger.exception(
            "Unhandled API exception",
            extra={
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
