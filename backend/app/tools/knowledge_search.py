"""Knowledge search tool backed by the hybrid retriever."""

from app.models.documents import RetrievedDocument
from app.retrieval.base import MetadataFilter
from app.retrieval.vector_store import retrieve_relevant_context
from app.tools.permissions import check_tool_permission


async def knowledge_search_tool(
    query: str,
    role: str,
    top_k: int = 3,
    metadata_filter: MetadataFilter | None = None,
) -> list[RetrievedDocument]:
    """Search approved knowledge chunks for the caller's role."""

    check_tool_permission("knowledge_search_tool", role)
    return await retrieve_relevant_context(
        query=query,
        role=role,
        limit=top_k,
        metadata_filter=metadata_filter,
    )
