"""Placeholder document ingestion script.

Later this script will chunk documents, create embeddings, and upsert vectors
into Pinecone. For now it lists discovered files so the workflow is visible.
"""

from pathlib import Path

from app.services.document_service import discover_documents


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "sample_docs"


def main() -> None:
    """Print documents that would be ingested."""

    documents = discover_documents(SAMPLE_DOCS)
    if not documents:
        print("No documents found. Run scripts/seed_mock_docs.py first.")
        return

    for document in documents:
        print(f"Would ingest: {document}")


if __name__ == "__main__":
    main()
