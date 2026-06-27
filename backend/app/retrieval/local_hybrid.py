"""Local hybrid retrieval over JSONL document chunks.

This module combines a dense score from an embedding provider with a sparse
BM25 keyword score. Pinecone can replace this later while preserving the same
search-facing behavior.
"""

import asyncio
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.models.documents import DocumentChunk, RetrievedDocument
from app.retrieval.base import MetadataFilter
from app.retrieval.embeddings import (
    EmbeddingProvider,
    FallbackEmbeddingProvider,
    HashEmbeddingProvider,
    cosine_similarity,
    tokenize,
)
from app.security.guardrails import ROLE_ACCESS_LEVELS


def load_chunks(jsonl_path: Path) -> list[DocumentChunk]:
    """Load document chunks from local JSONL storage."""

    if not jsonl_path.exists():
        return []

    chunks: list[DocumentChunk] = []
    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                chunks.append(DocumentChunk(**json.loads(line)))

    return chunks


def filter_chunks_by_role(chunks: list[DocumentChunk], role: str) -> list[DocumentChunk]:
    """Apply simple role-to-access-level metadata filtering."""

    allowed_access_levels = ROLE_ACCESS_LEVELS.get(role.lower(), set())
    return [chunk for chunk in chunks if chunk.access_level.lower() in allowed_access_levels]


def filter_chunks_by_metadata(
    chunks: list[DocumentChunk],
    metadata_filter: MetadataFilter | None,
) -> list[DocumentChunk]:
    """Apply exact-match metadata filters to local chunks."""

    if not metadata_filter:
        return chunks

    filtered_chunks = chunks
    for field_name, expected_value in metadata_filter.items():
        expected_values = expected_value if isinstance(expected_value, list) else [expected_value]
        allowed_values = {str(value).lower() for value in expected_values}
        filtered_chunks = [
            chunk
            for chunk in filtered_chunks
            if str(getattr(chunk, field_name, "")).lower() in allowed_values
        ]

    return filtered_chunks


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to 0..1 for hybrid ranking."""

    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0 if max_score > 0 else 0.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


class LocalHybridRetriever:
    """Search local JSONL chunks with dense and sparse ranking signals."""

    def __init__(
        self,
        jsonl_path: Path,
        embedding_provider: EmbeddingProvider | None = None,
        alpha: float = 0.6,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.jsonl_path = jsonl_path
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.alpha = alpha

    async def search(
        self,
        query: str,
        role: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedDocument]:
        """Return top_k chunks with citations after metadata filtering."""

        if top_k <= 0:
            return []

        permitted_chunks = filter_chunks_by_metadata(
            filter_chunks_by_role(load_chunks(self.jsonl_path), role),
            metadata_filter,
        )
        if not permitted_chunks:
            return []

        sparse_scores = self._sparse_scores(query, permitted_chunks)
        dense_scores = await self._dense_scores(query, permitted_chunks)

        ranked_results = []
        for chunk, dense_score, sparse_score in zip(
            permitted_chunks,
            dense_scores,
            sparse_scores,
            strict=True,
        ):
            hybrid_score = (self.alpha * dense_score) + ((1 - self.alpha) * sparse_score)
            ranked_results.append((hybrid_score, dense_score, sparse_score, chunk))

        ranked_results.sort(key=lambda result: result[0], reverse=True)

        return [
            RetrievedDocument(
                id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                title=chunk.title,
                snippet=chunk.text,
                score=round(hybrid_score, 6),
                dense_score=round(dense_score, 6),
                sparse_score=round(sparse_score, 6),
                source_file=chunk.source_file,
                department=chunk.department,
                document_type=chunk.document_type,
                access_level=chunk.access_level,
                created_date=chunk.created_date,
            )
            for hybrid_score, dense_score, sparse_score, chunk in ranked_results[:top_k]
        ]

    def _sparse_scores(self, query: str, chunks: list[DocumentChunk]) -> list[float]:
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        query_tokens = tokenize(query)
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = [float(score) for score in bm25.get_scores(query_tokens)]

        if raw_scores and max(raw_scores) <= 0:
            query_terms = set(query_tokens)
            raw_scores = [
                float(sum(1 for token in document_tokens if token in query_terms))
                for document_tokens in tokenized_corpus
            ]

        return normalize_scores(raw_scores)

    async def _dense_scores(self, query: str, chunks: list[DocumentChunk]) -> list[float]:
        document_texts = [chunk.text for chunk in chunks]
        if isinstance(self.embedding_provider, FallbackEmbeddingProvider):
            query_vector, chunk_vectors = (
                await self.embedding_provider.embed_query_and_documents(
                    query,
                    document_texts,
                )
            )
            return [
                cosine_similarity(query_vector, chunk_vector)
                for chunk_vector in chunk_vectors
            ]

        query_vectors, chunk_vectors = await asyncio.gather(
            self.embedding_provider.embed_texts([query], task="query"),
            self.embedding_provider.embed_texts(
                document_texts,
                task="document",
            ),
        )
        query_vector = query_vectors[0]
        return [cosine_similarity(query_vector, chunk_vector) for chunk_vector in chunk_vectors]
