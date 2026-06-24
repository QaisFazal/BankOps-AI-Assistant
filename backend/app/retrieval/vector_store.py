"""Retrieval abstraction.

This module chooses a retriever implementation while keeping callers on one
interface. Pinecone is optional; local hybrid retrieval is the safe fallback.
"""

import logging
from pathlib import Path

from app.config import get_settings
from app.models.documents import RetrievedDocument
from app.retrieval.base import MetadataFilter, Retriever
from app.retrieval.local_hybrid import LocalHybridRetriever
from app.retrieval.pinecone_hybrid import PineconeHybridRetriever


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSONL_PATH = ROOT / "data" / "document_chunks.jsonl"


def build_retriever(jsonl_path: Path = DEFAULT_JSONL_PATH) -> Retriever:
    """Build the configured retriever with local fallback."""

    settings = get_settings()
    local_retriever = LocalHybridRetriever(jsonl_path=jsonl_path, alpha=settings.retrieval_alpha)

    if settings.retrieval_backend != "pinecone":
        return local_retriever

    if not settings.pinecone_api_key:
        logger.warning("Pinecone backend requested without PINECONE_API_KEY; using local retrieval")
        return local_retriever

    try:
        return PineconeHybridRetriever(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            environment=settings.environment,
            namespace=settings.pinecone_namespace,
            namespace_mode=settings.pinecone_namespace_mode,
            alpha=settings.retrieval_alpha,
            sparse_dimensions=settings.pinecone_sparse_dimensions,
            fallback_retriever=local_retriever,
        )
    except Exception:
        logger.exception("Could not initialize Pinecone retriever; using local retrieval")
        return local_retriever


async def retrieve_relevant_context(
    query: str,
    role: str,
    limit: int = 3,
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    metadata_filter: MetadataFilter | None = None,
) -> list[RetrievedDocument]:
    """Return relevant chunks from the configured retriever."""

    retriever = build_retriever(jsonl_path=jsonl_path)
    return await retriever.search(
        query=query,
        role=role,
        top_k=limit,
        metadata_filter=metadata_filter,
    )
