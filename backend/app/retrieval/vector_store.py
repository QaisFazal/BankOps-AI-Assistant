"""Pinecone retrieval abstraction.

Keep application code talking to this module instead of importing Pinecone
directly everywhere. That makes it easier to swap in a mock store for tests or
an in-memory vector database for local demos.
"""

from app.models.documents import RetrievedDocument


def retrieve_relevant_context(query: str, limit: int = 3) -> list[RetrievedDocument]:
    """Return placeholder documents related to the query."""

    _ = (query, limit)
    return [
        RetrievedDocument(
            id="mock-policy-001",
            title="Mock Enterprise AI Policy",
            snippet="Placeholder source returned until Pinecone ingestion is wired up.",
            score=0.0,
        )
    ]
