"""LangSmith tracing helpers.

The app can run without LangSmith credentials. When tracing is disabled or the
API key is missing, these helpers become no-ops so local development and tests
stay simple.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import os
from typing import Any

from langsmith.run_helpers import trace

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Prepare LangSmith environment variables for traced runs."""

    remote_tracing_enabled = tracing_enabled(settings)
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if remote_tracing_enabled else "false"
    # Some LangChain/LangSmith integrations still read the older variable name.
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if remote_tracing_enabled else "false"

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key


def tracing_enabled(settings: Settings | None = None) -> bool:
    """Return whether remote LangSmith tracing should be attempted."""

    active_settings = settings or get_settings()
    return bool(active_settings.langsmith_tracing and active_settings.langsmith_api_key)


def build_run_metadata(
    *,
    user_id: str | None = None,
    role: str | None = None,
    session_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create consistent metadata for LangSmith run filtering."""

    metadata: dict[str, Any] = {}
    if user_id:
        metadata["user_id"] = user_id
    if role:
        metadata["role"] = role
    if session_id:
        metadata["session_id"] = session_id
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


@contextmanager
def trace_run(
    name: str,
    *,
    run_type: str,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any | None]:
    """Start a LangSmith run when enabled, otherwise yield ``None``."""

    settings = get_settings()
    if not tracing_enabled(settings):
        yield None
        return

    trace_context = trace(
        name,
        run_type=run_type,
        inputs=inputs or {},
        metadata=metadata or {},
        tags=tags or [],
        project_name=settings.langsmith_project,
    )
    try:
        run = trace_context.__enter__()
    except Exception:
        logger.warning("Could not start LangSmith trace for run %s", name, exc_info=True)
        yield None
        return

    try:
        yield run
    except BaseException as exc:
        try:
            trace_context.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.warning("Could not finish failed LangSmith trace for run %s", name, exc_info=True)
        raise
    else:
        try:
            trace_context.__exit__(None, None, None)
        except Exception:
            logger.warning("Could not finish LangSmith trace for run %s", name, exc_info=True)


def set_trace_outputs(run: Any | None, outputs: dict[str, Any]) -> None:
    """Attach summarized outputs to a LangSmith run if tracing is active."""

    if run is None:
        return

    try:
        run.end(outputs=outputs)
    except Exception:
        logger.warning("Could not attach outputs to LangSmith run", exc_info=True)
