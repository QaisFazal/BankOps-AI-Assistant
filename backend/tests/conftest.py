"""Shared pytest isolation for external AI and retrieval services."""

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def disable_external_services(monkeypatch):
    """Keep unit tests deterministic and prevent accidental billable API calls."""

    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("RETRIEVAL_BACKEND", "local")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
