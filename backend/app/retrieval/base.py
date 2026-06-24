"""Shared retrieval interface."""

from typing import Protocol

from app.models.documents import RetrievedDocument


MetadataFilter = dict[str, str | list[str]]


class Retriever(Protocol):
    """Common interface for local and Pinecone-backed retrievers."""

    async def search(
        self,
        query: str,
        role: str,
        top_k: int = 5,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievedDocument]:
        """Return ranked chunks with document attribution."""
