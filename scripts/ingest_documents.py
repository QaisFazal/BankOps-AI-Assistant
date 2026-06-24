"""Ingest local markdown documents into JSONL chunk storage."""

from pathlib import Path

from app.services.document_service import ingest_documents_to_jsonl


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "sample_docs"
OUTPUT_PATH = ROOT / "data" / "document_chunks.jsonl"


def main() -> None:
    """Parse sample docs and write local JSONL chunks."""

    chunks = ingest_documents_to_jsonl(SAMPLE_DOCS, OUTPUT_PATH)
    if not chunks:
        print("No documents found. Run scripts/seed_mock_docs.py first.")
        return

    print(f"Wrote {len(chunks)} chunk(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
