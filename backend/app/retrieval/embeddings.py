"""Embedding providers shared by local and Pinecone retrieval."""

import asyncio
import hashlib
import logging
import math
import re
from typing import Any, Literal, Protocol

from app.config import get_settings


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
EmbeddingTask = Literal["document", "query"]
logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Minimal async interface for dense embedding providers."""

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        """Return one vector per input text."""


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval."""

    return TOKEN_PATTERN.findall(text.lower())


class HashEmbeddingProvider:
    """Small deterministic embedding provider for local development.

    This is not semantically rich like a model embedding, but it gives the app a
    dense-vector scoring path now and keeps the implementation offline.
    """

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        """Embed text into normalized hashed bag-of-words vectors."""

        _ = task
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]


class GeminiEmbeddingProvider:
    """Generate semantic retrieval vectors with the Google GenAI SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")

        self.model = model
        self.dimensions = dimensions
        self.timeout_seconds = timeout_seconds
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        """Return one normalized Gemini vector for each supplied text."""

        if not texts:
            return []
        if task not in {"document", "query"}:
            raise ValueError(f"Unsupported embedding task: {task}.")

        from google.genai import types

        prepared_texts = [self._prepare_text(text, task) for text in texts]
        contents = [
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in prepared_texts
        ]
        response = await asyncio.wait_for(
            self.client.aio.models.embed_content(
                model=self.model,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimensions,
                ),
            ),
            timeout=self.timeout_seconds,
        )
        vectors = [list(embedding.values or []) for embedding in response.embeddings or []]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs."
            )
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(
                f"Gemini returned an embedding with an unexpected dimension; "
                f"expected {self.dimensions}."
            )
        return vectors

    @staticmethod
    def _prepare_text(text: str, task: EmbeddingTask) -> str:
        """Apply Gemini's recommended asymmetric retrieval formatting."""

        if task == "query":
            return f"task: search result | query: {text}"
        return f"title: none | text: {text}"


class FallbackEmbeddingProvider:
    """Use a local provider when the remote embedding service is unavailable."""

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        try:
            return await self.primary.embed_texts(texts, task=task)
        except Exception as exc:
            self._log_fallback(exc, task)
            return await self.fallback.embed_texts(texts, task=task)

    async def embed_query_and_documents(
        self,
        query: str,
        documents: list[str],
    ) -> tuple[list[float], list[list[float]]]:
        """Keep query and document vectors in the same embedding space."""

        try:
            query_vectors, document_vectors = await asyncio.gather(
                self.primary.embed_texts([query], task="query"),
                self.primary.embed_texts(documents, task="document"),
            )
            return query_vectors[0], document_vectors
        except Exception as exc:
            self._log_fallback(exc, "query_and_documents")
            query_vectors, document_vectors = await asyncio.gather(
                self.fallback.embed_texts([query], task="query"),
                self.fallback.embed_texts(documents, task="document"),
            )
            return query_vectors[0], document_vectors

    @staticmethod
    def _log_fallback(exc: Exception, task: str) -> None:
        logger.exception(
            "Remote embedding failed; using local hash embeddings",
            extra={
                "component": "retrieval",
                "operation": "embed_texts",
                "error_type": type(exc).__name__,
                "fallback": "hash_embedding_provider",
                "embedding_task": task,
            },
        )


def build_embedding_provider(*, allow_hash_fallback: bool = True) -> EmbeddingProvider:
    """Build the configured provider, optionally retaining an offline fallback."""

    settings = get_settings()
    hash_provider = HashEmbeddingProvider(dimensions=settings.embedding_dimensions)
    provider_name = settings.embedding_provider.strip().lower()

    if provider_name == "hash":
        return hash_provider
    if provider_name != "gemini":
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}.")
    if not settings.gemini_api_key:
        if not allow_hash_fallback:
            raise ValueError("GEMINI_API_KEY is required for Pinecone ingestion.")
        logger.warning(
            "Gemini embedding key missing; using local hash embeddings",
            extra={
                "component": "retrieval",
                "operation": "build_embedding_provider",
                "fallback": "hash_embedding_provider",
            },
        )
        return hash_provider

    gemini_provider = GeminiEmbeddingProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.tool_timeout_seconds,
    )
    if not allow_hash_fallback:
        return gemini_provider
    return FallbackEmbeddingProvider(gemini_provider, hash_provider)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two normalized vectors."""

    if not left or not right:
        return 0.0

    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
