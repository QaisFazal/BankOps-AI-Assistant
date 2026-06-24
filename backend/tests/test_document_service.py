"""Tests for local markdown document ingestion."""

import json

from app.services.document_service import (
    ingest_documents_to_jsonl,
    parse_markdown_with_metadata,
    split_text_into_chunks,
)


def test_parse_markdown_with_yaml_frontmatter(tmp_path) -> None:
    """Metadata should come from YAML frontmatter and body should stay markdown."""

    document = tmp_path / "incident.md"
    document.write_text(
        """---
department: payments
document_type: incident
access_level: internal
created_date: 2025-01-01
title: Payment Incident
source_file: incidents/payment-incident.md
---

# Payment Incident

The payment gateway timed out.
""",
        encoding="utf-8",
    )

    metadata, body = parse_markdown_with_metadata(document)

    assert metadata.department == "payments"
    assert metadata.document_type == "incident"
    assert metadata.access_level == "internal"
    assert metadata.created_date == "2025-01-01"
    assert metadata.title == "Payment Incident"
    assert metadata.source_file == "incidents/payment-incident.md"
    assert body.startswith("# Payment Incident")


def test_split_text_into_chunks_keeps_chunks_under_limit() -> None:
    """Chunking should group paragraphs without exceeding the target size."""

    text = "First paragraph about payments.\n\nSecond paragraph about cards.\n\nThird paragraph."

    chunks = split_text_into_chunks(text, max_chars=50)

    assert chunks == [
        "First paragraph about payments.",
        "Second paragraph about cards.\n\nThird paragraph.",
    ]
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_ingest_documents_to_jsonl_writes_metadata_and_chunk_ids(tmp_path) -> None:
    """Ingestion should write one JSON object per chunk with required metadata."""

    source_dir = tmp_path / "sample_docs"
    source_dir.mkdir()
    document = source_dir / "runbook.md"
    document.write_text(
        """---
department: cards
document_type: runbook
access_level: internal
created_date: 2025-02-01
title: Card Restart Runbook
source_file: runbooks/card-restart.md
---

# Card Restart Runbook

Restart one pod group at a time.
""",
        encoding="utf-8",
    )
    output_path = tmp_path / "document_chunks.jsonl"

    chunks = ingest_documents_to_jsonl(source_dir, output_path, max_chars=200)
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(chunks) == 1
    assert len(records) == 1
    assert records[0]["chunk_id"].endswith("-0000")
    assert records[0]["source_file"] == "runbooks/card-restart.md"
    assert records[0]["title"] == "Card Restart Runbook"
    assert records[0]["department"] == "cards"
    assert records[0]["document_type"] == "runbook"
    assert records[0]["access_level"] == "internal"
    assert records[0]["created_date"] == "2025-02-01"
