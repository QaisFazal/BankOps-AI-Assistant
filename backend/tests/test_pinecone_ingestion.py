"""Tests for Pinecone document upsert helpers."""

import asyncio
import json

from app.models.documents import DocumentChunk
from app.retrieval.embeddings import HashEmbeddingProvider
from app.services import pinecone_ingestion
from app.services.pinecone_ingestion import (
    build_chunk_metadata,
    build_sparse_vector,
    resolve_upsert_namespace,
    upsert_chunks_to_pinecone,
)


class FakeIndex:
    """Capture upsert calls for tests."""

    def __init__(self) -> None:
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)
        return {"upserted_count": len(kwargs["vectors"])}


def make_chunk(
    chunk_id: str = "chunk-1",
    department: str = "payments",
    text: str = "Payment gateway timeout incident.",
) -> DocumentChunk:
    """Build one test document chunk."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        text=text,
        source_file="incidents/payment.md",
        title="Payment Incident",
        department=department,
        document_type="incident",
        access_level="internal",
        created_date="2025-01-01",
    )


def test_build_sparse_vector_matches_expected_shape() -> None:
    """Sparse vectors should contain sorted indices and weighted values."""

    sparse_vector = build_sparse_vector(
        "payment payment outage",
        sparse_dimensions=100_000,
    )

    assert sorted(sparse_vector["indices"]) == sparse_vector["indices"]
    assert len(sparse_vector["indices"]) == len(sparse_vector["values"])
    assert all(value > 0 for value in sparse_vector["values"])
    assert max(sparse_vector["values"]) == 2.0


def test_resolve_upsert_namespace_supports_environment_and_department() -> None:
    """Namespace selection should match retrieval adapter behavior."""

    chunk = make_chunk(department="cards")

    assert (
        resolve_upsert_namespace(
            chunk,
            environment="local",
            namespace="local",
            namespace_mode="environment",
        )
        == "local"
    )
    assert (
        resolve_upsert_namespace(
            chunk,
            environment="local",
            namespace="local",
            namespace_mode="department",
        )
        == "local-cards"
    )


def test_build_chunk_metadata_includes_attribution_fields() -> None:
    """Metadata should contain fields needed for retrieval and citations."""

    metadata = build_chunk_metadata(make_chunk())

    assert metadata["document_id"] == "doc-1"
    assert metadata["text"] == "Payment gateway timeout incident."
    assert metadata["source_file"] == "incidents/payment.md"
    assert metadata["access_level"] == "internal"


def test_upsert_chunks_to_pinecone_groups_vectors_by_namespace(tmp_path, monkeypatch) -> None:
    """Upsert should group vectors into the configured Pinecone namespaces."""

    chunks = [
        make_chunk(chunk_id="payments-1", department="payments"),
        make_chunk(chunk_id="cards-1", department="cards", text="Card authorization latency."),
    ]
    jsonl_path = tmp_path / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk.model_dump()) + "\n")

    monkeypatch.setattr(
        pinecone_ingestion,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {
                "pinecone_api_key": "test-key",
                "pinecone_index_name": "test-index",
                "environment": "local",
                "pinecone_namespace": "local",
                "pinecone_namespace_mode": "department",
                "retrieval_alpha": 0.6,
                "pinecone_sparse_dimensions": 100_000,
            },
        )(),
    )
    fake_index = FakeIndex()

    upserted_count = asyncio.run(
        upsert_chunks_to_pinecone(
            jsonl_path=jsonl_path,
            batch_size=1,
            embedding_provider=HashEmbeddingProvider(dimensions=3),
            index=fake_index,
        )
    )

    namespaces = {call["namespace"] for call in fake_index.calls}

    assert upserted_count == 2
    assert namespaces == {"local-payments", "local-cards"}
    assert fake_index.calls[0]["vectors"][0]["metadata"]["text"]
    assert len(fake_index.calls[0]["vectors"][0]["values"]) == 3
    assert "sparse_values" in fake_index.calls[0]["vectors"][0]
