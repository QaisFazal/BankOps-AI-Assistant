"""Retrieval abstraction.

This module chooses a retriever implementation while keeping callers on one
interface. Pinecone is optional; local hybrid retrieval is the safe fallback.
"""

import logging
from pathlib import Path

from app.config import get_settings
from app.models.documents import RetrievedDocument
from app.observability.tracing import build_run_metadata, set_trace_outputs, trace_run
from app.retrieval.base import MetadataFilter, Retriever
from app.retrieval.embeddings import build_embedding_provider
from app.retrieval.local_hybrid import LocalHybridRetriever
from app.retrieval.pinecone_hybrid import PineconeHybridRetriever


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSONL_PATH = ROOT / "data" / "document_chunks.jsonl"


def build_retriever(jsonl_path: Path = DEFAULT_JSONL_PATH) -> Retriever:
    """Build the configured retriever with local fallback."""

    settings = get_settings()
    local_retriever = LocalHybridRetriever(
        jsonl_path=jsonl_path,
        embedding_provider=build_embedding_provider(allow_hash_fallback=True),
        alpha=settings.retrieval_alpha,
    )

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
            embedding_provider=build_embedding_provider(allow_hash_fallback=False),
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
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[RetrievedDocument]:
    """Return relevant chunks from the configured retriever."""

    retriever = build_retriever(jsonl_path=jsonl_path)
    settings = get_settings()
    metadata = build_run_metadata(
        user_id=user_id,
        role=role,
        session_id=session_id,
        retrieval_backend=settings.retrieval_backend,
        retriever_type=type(retriever).__name__,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.gemini_embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )
    with trace_run(
        "hybrid_retrieval",
        run_type="retriever",
        inputs={"query": query, "top_k": limit, "metadata_filter": metadata_filter},
        metadata=metadata,
        tags=["retrieval", settings.retrieval_backend],
    ) as run:
        results = await retriever.search(
            query=query,
            role=role,
            top_k=limit,
            metadata_filter=metadata_filter,
        )
        set_trace_outputs(
            run,
            {
                "result_count": len(results),
                "sources": [
                    {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "title": result.title,
                        "source_file": result.source_file,
                        "score": result.score,
                    }
                    for result in results
                ],
            },
        )
        return results
