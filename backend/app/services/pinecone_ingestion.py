"""Pinecone upsert support for local document chunks."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from pinecone import Pinecone

from app.config import get_settings
from app.models.documents import DocumentChunk
from app.retrieval.embeddings import HashEmbeddingProvider, tokenize


DEFAULT_BATCH_SIZE = 100


def load_document_chunks(jsonl_path: Path) -> list[DocumentChunk]:
    """Load JSONL chunks produced by the local ingestion pipeline."""

    if not jsonl_path.exists():
        return []

    chunks: list[DocumentChunk] = []
    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                chunks.append(DocumentChunk(**json.loads(line)))

    return chunks


def build_sparse_vector(
    text: str,
    *,
    alpha: float,
    sparse_dimensions: int,
) -> dict[str, list[int] | list[float]]:
    """Build the same sparse vector shape used by Pinecone query calls."""

    token_counts: dict[int, float] = {}
    for token in tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % sparse_dimensions
        token_counts[index] = token_counts.get(index, 0.0) + 1.0

    indices = sorted(token_counts)
    return {
        "indices": indices,
        "values": [token_counts[index] * (1 - alpha) for index in indices],
    }


def resolve_upsert_namespace(
    chunk: DocumentChunk,
    *,
    environment: str,
    namespace: str,
    namespace_mode: str,
) -> str:
    """Resolve the namespace where a chunk should be written."""

    if namespace_mode == "department":
        return f"{environment}-{chunk.department}"

    return namespace or environment


def build_chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    """Return Pinecone metadata needed for retrieval and attribution."""

    return {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "source_file": chunk.source_file,
        "title": chunk.title,
        "department": chunk.department,
        "document_type": chunk.document_type,
        "access_level": chunk.access_level,
        "created_date": chunk.created_date,
    }


async def build_pinecone_vector(
    chunk: DocumentChunk,
    *,
    embedding_provider: HashEmbeddingProvider,
    alpha: float,
    sparse_dimensions: int,
) -> dict[str, Any]:
    """Build one Pinecone hybrid vector payload."""

    dense_vector = (await embedding_provider.embed_texts([chunk.text]))[0]
    return {
        "id": chunk.chunk_id,
        "values": [value * alpha for value in dense_vector],
        "sparse_values": build_sparse_vector(
            chunk.text,
            alpha=alpha,
            sparse_dimensions=sparse_dimensions,
        ),
        "metadata": build_chunk_metadata(chunk),
    }


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split items into fixed-size batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


async def upsert_chunks_to_pinecone(
    *,
    jsonl_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    index: Any | None = None,
) -> int:
    """Read local chunks and upsert them into the configured Pinecone index."""

    settings = get_settings()
    if not settings.pinecone_api_key and index is None:
        raise ValueError("PINECONE_API_KEY is required to upsert chunks into Pinecone.")

    chunks = load_document_chunks(jsonl_path)
    if not chunks:
        return 0

    pinecone_index = index or Pinecone(api_key=settings.pinecone_api_key).Index(
        settings.pinecone_index_name
    )
    embedding_provider = HashEmbeddingProvider()
    upserted_count = 0

    vectors_by_namespace: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        namespace = resolve_upsert_namespace(
            chunk,
            environment=settings.environment,
            namespace=settings.pinecone_namespace,
            namespace_mode=settings.pinecone_namespace_mode,
        )
        vectors_by_namespace.setdefault(namespace, []).append(
            await build_pinecone_vector(
                chunk,
                embedding_provider=embedding_provider,
                alpha=settings.retrieval_alpha,
                sparse_dimensions=settings.pinecone_sparse_dimensions,
            )
        )

    for namespace, vectors in vectors_by_namespace.items():
        for vector_batch in batched(vectors, batch_size):
            maybe_response = pinecone_index.upsert(vectors=vector_batch, namespace=namespace)
            if asyncio.iscoroutine(maybe_response):
                await maybe_response
            upserted_count += len(vector_batch)

    return upserted_count
