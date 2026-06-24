"""LangSmith tracing setup placeholder."""

from app.config import Settings


def configure_tracing(settings: Settings) -> None:
    """Prepare LangSmith tracing when credentials are available."""

    if not settings.langsmith_tracing:
        return

    # Later: configure LangSmith callbacks or environment variables here.
