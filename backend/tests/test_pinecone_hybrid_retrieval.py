"""Tests for the Pinecone hybrid retrieval adapter."""

import asyncio

from app.models.documents import RetrievedDocument
from app.retrieval.pinecone_hybrid import PineconeHybridRetriever


class FakeEmbeddingProvider:
    """Return a tiny deterministic vector for tests."""

    async def embed_texts(
        self,
        texts: list[str],
        *,
        task: str = "document",
    ) -> list[list[float]]:
        _ = task
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeIndex:
    """Capture Pinecone query arguments and return one match."""

    def __init__(self) -> None:
        self.last_query = None

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "matches": [
                {
                    "id": "chunk-0000",
                    "score": 0.91,
                    "metadata": {
                        "document_id": "doc-1",
                        "text": "Card authorization latency runbook content.",
                        "source_file": "runbooks/card-auth-latency.md",
                        "title": "Card Authorization Latency",
                        "department": "cards",
                        "document_type": "runbook",
                        "access_level": "internal",
                        "created_date": "2025-01-01",
                    },
                }
            ]
        }


class FailingIndex:
    """Simulate Pinecone being unavailable."""

    def query(self, **kwargs):
        _ = kwargs
        raise RuntimeError("pinecone unavailable")


class FakeFallbackRetriever:
    """Return a known result when Pinecone fails."""

    async def search(self, query, role, top_k=5, metadata_filter=None):
        _ = (query, role, top_k, metadata_filter)
        return [
            RetrievedDocument(
                id="fallback-0000",
                title="Fallback Result",
                snippet="Returned by local fallback.",
                score=0.5,
            )
        ]


def test_pinecone_search_uses_namespace_filter_and_attribution() -> None:
    """Pinecone adapter should send filters and map document attribution."""

    fake_index = FakeIndex()
    retriever = PineconeHybridRetriever(
        api_key="test-key",
        index_name="test-index",
        environment="local",
        namespace="local",
        namespace_mode="department",
        embedding_provider=FakeEmbeddingProvider(),
        alpha=0.7,
        index=fake_index,
    )

    results = asyncio.run(
        retriever.search(
            "card authorization latency",
            role="analyst",
            top_k=3,
            metadata_filter={"department": "cards", "document_type": "runbook"},
        )
    )

    assert fake_index.last_query["namespace"] == "local-cards"
    assert fake_index.last_query["filter"]["access_level"] == {
        "$in": ["confidential", "internal"]
    }
    assert fake_index.last_query["filter"]["department"] == {"$eq": "cards"}
    assert fake_index.last_query["filter"]["document_type"] == {"$eq": "runbook"}
    assert fake_index.last_query["include_metadata"] is True
    assert fake_index.last_query["vector"] == [0.7, 0.0, 0.0]
    assert results[0].chunk_id == "chunk-0000"
    assert results[0].document_id == "doc-1"
    assert results[0].source_file == "runbooks/card-auth-latency.md"
    assert results[0].title == "Card Authorization Latency"


def test_pinecone_search_falls_back_to_local_retriever() -> None:
    """Pinecone failures should gracefully return fallback results."""

    retriever = PineconeHybridRetriever(
        api_key="test-key",
        index_name="test-index",
        environment="local",
        namespace="local",
        embedding_provider=FakeEmbeddingProvider(),
        fallback_retriever=FakeFallbackRetriever(),
        index=FailingIndex(),
    )

    results = asyncio.run(retriever.search("anything", role="viewer", top_k=1))

    assert results[0].id == "fallback-0000"
