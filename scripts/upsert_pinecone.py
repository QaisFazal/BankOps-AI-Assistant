"""Upsert local document chunks into Pinecone."""

import asyncio
from pathlib import Path

from app.services.pinecone_ingestion import upsert_chunks_to_pinecone


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "document_chunks.jsonl"


async def main() -> None:
    """Load local chunks and upsert them into the configured Pinecone index."""

    upserted_count = await upsert_chunks_to_pinecone(jsonl_path=CHUNKS_PATH)
    if upserted_count == 0:
        print("No chunks found. Run scripts/ingest_documents.py first.")
        return

    print(f"Upserted {upserted_count} chunk(s) into Pinecone.")


if __name__ == "__main__":
    asyncio.run(main())
