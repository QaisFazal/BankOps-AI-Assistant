"""Tests for semantic embedding providers and local fallback behavior."""

import asyncio
from types import SimpleNamespace

from app.retrieval.embeddings import (
    FallbackEmbeddingProvider,
    GeminiEmbeddingProvider,
    HashEmbeddingProvider,
)


class FakeAsyncModels:
    """Capture Gemini embedding requests and return fixed vectors."""

    def __init__(self) -> None:
        self.last_call = None

    async def embed_content(self, **kwargs):
        self.last_call = kwargs
        count = len(kwargs["contents"])
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.5, 0.5, 0.5]) for _ in range(count)]
        )


class FailingEmbeddingProvider:
    """Simulate an unavailable remote embedding service."""

    async def embed_texts(self, texts, *, task="document"):
        _ = (texts, task)
        raise RuntimeError("embedding provider unavailable")


def test_gemini_embedding_provider_returns_one_vector_per_text() -> None:
    """Gemini should receive separate contents and return configured dimensions."""

    models = FakeAsyncModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        model="gemini-embedding-2",
        dimensions=3,
        client=client,
    )

    vectors = asyncio.run(
        provider.embed_texts(["payment outage", "card latency"], task="document")
    )

    assert vectors == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    assert models.last_call["model"] == "gemini-embedding-2"
    assert models.last_call["config"].output_dimensionality == 3


def test_gemini_embedding_provider_formats_query_and_document_differently() -> None:
    """Asymmetric retrieval formatting should distinguish questions from documents."""

    assert GeminiEmbeddingProvider._prepare_text("payment outage", "query") == (
        "task: search result | query: payment outage"
    )
    assert GeminiEmbeddingProvider._prepare_text("payment outage", "document") == (
        "title: none | text: payment outage"
    )


def test_embedding_provider_falls_back_to_local_hash_vectors() -> None:
    """Local retrieval should remain available when Gemini embeddings fail."""

    provider = FallbackEmbeddingProvider(
        FailingEmbeddingProvider(),
        HashEmbeddingProvider(dimensions=4),
    )

    vectors = asyncio.run(provider.embed_texts(["payment outage"], task="query"))

    assert len(vectors) == 1
    assert len(vectors[0]) == 4


def test_retrieval_fallback_keeps_query_and_documents_in_one_vector_space() -> None:
    """A remote failure should recompute both sides with the fallback provider."""

    provider = FallbackEmbeddingProvider(
        FailingEmbeddingProvider(),
        HashEmbeddingProvider(dimensions=4),
    )

    query_vector, document_vectors = asyncio.run(
        provider.embed_query_and_documents("payment outage", ["gateway failure"])
    )

    assert len(query_vector) == 4
    assert len(document_vectors) == 1
    assert len(document_vectors[0]) == 4
