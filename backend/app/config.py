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

    openai_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-lead-assistant"
    langsmith_tracing: bool = False

    pinecone_api_key: str | None = None
    pinecone_index_name: str = "ai-lead-assistant"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Cache settings so every import does not reload environment variables."""

    return Settings()
