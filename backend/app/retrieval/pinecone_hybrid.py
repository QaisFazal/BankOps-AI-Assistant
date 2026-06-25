"""Pinecone-backed hybrid retrieval adapter."""

import hashlib
import logging
from typing import Any

from pinecone import Pinecone

from app.models.documents import RetrievedDocument
from app.retrieval.base import MetadataFilter, Retriever
from app.retrieval.embeddings import EmbeddingProvider, HashEmbeddingProvider, tokenize
from app.security.guardrails import ROLE_ACCESS_LEVELS


logger = logging.getLogger(__name__)


class PineconeHybridRetriever:
    """Search Pinecone with dense and sparse query signals.

    The adapter assumes indexed vectors include metadata fields matching
    DocumentChunk: source_file, title, department, document_type, access_level,
    created_date, and text.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str,
        namespace: str,
        namespace_mode: str = "environment",
        embedding_provider: EmbeddingProvider | None = None,
        alpha: float = 0.6,
        sparse_dimensions: int = 100_000,
        fallback_retriever: Retriever | None = None,
        index: Any | None = None,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        self.index_name = index_name
        self.environment = environment
        self.namespace = namespace
        self.namespace_mode = namespace_mode
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.alpha = alpha
        self.sparse_dimensions = sparse_dimensions
        self.fallback_retriever = fallback_retriever
        self.index = index or Pinecone(api_key=api_key).Index(index_name)

    async def search(
        self,
        query: str,
        role: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedDocument]:
        """Search Pinecone, falling back to local retrieval if unavailable."""

        try:
            if top_k <= 0:
                return []

            dense_vector = [
                value * self.alpha
                for value in (await self.embedding_provider.embed_texts([query]))[0]
            ]
            response = self.index.query(
                vector=dense_vector,
                sparse_vector=self._build_sparse_vector(query),
                top_k=top_k,
                namespace=self._resolve_namespace(metadata_filter),
                filter=self._build_pinecone_filter(role, metadata_filter),
                include_metadata=True,
            )
            return self._response_to_documents(response)
        except Exception as exc:
            logger.exception(
                "Pinecone retrieval failed; falling back to local retrieval",
                extra={
                    "component": "retrieval",
                    "operation": "pinecone_hybrid_search",
                    "error_type": type(exc).__name__,
                    "fallback": "local_retriever" if self.fallback_retriever else "empty_results",
                    "role": role,
                    "retrieval_backend": "pinecone",
                },
            )
            if self.fallback_retriever is None:
                return []

            return await self.fallback_retriever.search(
                query=query,
                role=role,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

    def _resolve_namespace(self, metadata_filter: MetadataFilter | None) -> str:
        """Resolve namespace by environment or department."""

        if self.namespace_mode == "department" and metadata_filter:
            department = metadata_filter.get("department")
            if isinstance(department, str) and department:
                return f"{self.environment}-{department}"

        return self.namespace or self.environment

    def _build_pinecone_filter(
        self,
        role: str,
        metadata_filter: MetadataFilter | None,
    ) -> dict[str, Any]:
        """Create a Pinecone metadata filter from role and exact-match filters."""

        allowed_access_levels = sorted(ROLE_ACCESS_LEVELS.get(role.lower(), set()))
        pinecone_filter: dict[str, Any] = {
            "access_level": {"$in": allowed_access_levels},
        }

        for field_name, expected_value in (metadata_filter or {}).items():
            if field_name == "access_level":
                continue

            if isinstance(expected_value, list):
                pinecone_filter[field_name] = {"$in": expected_value}
            else:
                pinecone_filter[field_name] = {"$eq": expected_value}

        return pinecone_filter

    def _build_sparse_vector(self, query: str) -> dict[str, list[int] | list[float]]:
        """Build a deterministic sparse query vector for Pinecone hybrid search."""

        token_counts: dict[int, float] = {}
        for token in tokenize(query):
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.sparse_dimensions
            token_counts[index] = token_counts.get(index, 0.0) + 1.0

        indices = sorted(token_counts)
        return {
            "indices": indices,
            "values": [token_counts[index] * (1 - self.alpha) for index in indices],
        }

    def _response_to_documents(self, response: Any) -> list[RetrievedDocument]:
        """Map Pinecone query matches to app retrieval models."""

        matches = response.get("matches", []) if isinstance(response, dict) else response.matches
        documents: list[RetrievedDocument] = []
        for match in matches:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else match.metadata or {}
            chunk_id = match.get("id") if isinstance(match, dict) else match.id
            score = match.get("score", 0.0) if isinstance(match, dict) else match.score
            text = str(metadata.get("text", ""))

            documents.append(
                RetrievedDocument(
                    id=chunk_id,
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")) or None,
                    title=str(metadata.get("title", "")),
                    snippet=text[:500],
                    score=round(float(score or 0.0), 6),
                    source_file=str(metadata.get("source_file", "")),
                    department=str(metadata.get("department", "")),
                    document_type=str(metadata.get("document_type", "")),
                    access_level=str(metadata.get("access_level", "")),
                    created_date=str(metadata.get("created_date", "")),
                )
            )

        return documents
