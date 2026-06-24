"""Tests for local hybrid retrieval."""

import asyncio
import json

from app.models.documents import DocumentChunk
from app.retrieval.local_hybrid import LocalHybridRetriever


def write_chunks(path, chunks: list[DocumentChunk]) -> None:
    """Write test chunks to JSONL."""

    with path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk.model_dump()) + "\n")


def make_chunk(
    chunk_id: str,
    text: str,
    access_level: str,
    title: str,
    source_file: str,
) -> DocumentChunk:
    """Create a minimal chunk for retrieval tests."""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.rsplit("-", maxsplit=1)[0],
        chunk_index=0,
        text=text,
        source_file=source_file,
        title=title,
        department="payments",
        document_type="runbook",
        access_level=access_level,
        created_date="2025-01-01",
    )


def test_search_filters_chunks_by_role_access_level(tmp_path) -> None:
    """Viewer role should not receive confidential or restricted chunks."""

    jsonl_path = tmp_path / "chunks.jsonl"
    write_chunks(
        jsonl_path,
        [
            make_chunk(
                "internal-0000",
                "Payment outage checklist for internal support teams.",
                "internal",
                "Payment Outage Checklist",
                "runbooks/payment-outage.md",
            ),
            make_chunk(
                "restricted-0000",
                "Restricted card network failover procedure and approval path.",
                "restricted",
                "Card Network Failover",
                "runbooks/card-network-failover.md",
            ),
        ],
    )
    retriever = LocalHybridRetriever(jsonl_path=jsonl_path)

    viewer_results = asyncio.run(retriever.search("card network failover", role="viewer", top_k=5))
    admin_results = asyncio.run(
        retriever.search("card network failover", role="administrator", top_k=5)
    )

    assert [result.access_level for result in viewer_results] == ["internal"]
    assert any(result.access_level == "restricted" for result in admin_results)


def test_search_ranks_more_relevant_chunk_first(tmp_path) -> None:
    """Hybrid ranking should put the best matching chunk at the top."""

    jsonl_path = tmp_path / "chunks.jsonl"
    write_chunks(
        jsonl_path,
        [
            make_chunk(
                "payments-0000",
                "Payment queue backlog handling for ACH settlement delays.",
                "internal",
                "Payment Queue Backlog",
                "runbooks/payment-queue-backlog.md",
            ),
            make_chunk(
                "cards-0000",
                "Card authorization latency runbook for slow merchant approvals.",
                "internal",
                "Card Authorization Latency",
                "runbooks/card-auth-latency.md",
            ),
        ],
    )
    retriever = LocalHybridRetriever(jsonl_path=jsonl_path, alpha=0.5)

    results = asyncio.run(retriever.search("card authorization latency", role="viewer", top_k=2))

    assert results[0].source_file == "runbooks/card-auth-latency.md"
    assert results[0].score >= results[1].score
    assert results[0].dense_score > 0
    assert results[0].sparse_score > 0


def test_search_applies_metadata_filter(tmp_path) -> None:
    """Local retrieval should honor exact metadata filters."""

    jsonl_path = tmp_path / "chunks.jsonl"
    write_chunks(
        jsonl_path,
        [
            make_chunk(
                "payments-0000",
                "Payment queue backlog handling for ACH settlement delays.",
                "internal",
                "Payment Queue Backlog",
                "runbooks/payment-queue-backlog.md",
            ),
            make_chunk(
                "cards-0000",
                "Card authorization latency runbook for slow merchant approvals.",
                "internal",
                "Card Authorization Latency",
                "runbooks/card-auth-latency.md",
            ),
        ],
    )
    retriever = LocalHybridRetriever(jsonl_path=jsonl_path)

    results = asyncio.run(
        retriever.search(
            "authorization latency",
            role="viewer",
            top_k=5,
            metadata_filter={"source_file": "runbooks/card-auth-latency.md"},
        )
    )

    assert len(results) == 1
    assert results[0].source_file == "runbooks/card-auth-latency.md"
