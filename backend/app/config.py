"""Application settings loaded from environment variables.

For a beginner-friendly scaffold, keep configuration explicit and boring.
In production, secrets should come from a secret manager rather than a local
`.env` file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings shared across the backend."""

    app_name: str = "AI Lead Assistant"
    environment: str = "local"
    backend_cors_origins: str = "http://localhost:8501"
    rate_limit_capacity: int = 60
    rate_limit_refill_rate_per_second: float = 1.0
    tool_timeout_seconds: float = 30.0

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-lead-assistant"
    langsmith_tracing: bool = False

    retrieval_backend: str = "local"
    retrieval_alpha: float = 0.6

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "ai-lead-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "local"
    pinecone_namespace_mode: str = "environment"
    pinecone_sparse_dimensions: int = 100_000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Cache settings so every import does not reload environment variables."""

    return Settings()
