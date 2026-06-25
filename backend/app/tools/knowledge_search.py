"""Knowledge search tool backed by the hybrid retriever."""

import asyncio
import logging

from app.config import get_settings
from app.models.documents import RetrievedDocument
from app.observability.tracing import build_run_metadata, set_trace_outputs, trace_run
from app.retrieval.base import MetadataFilter
from app.retrieval.vector_store import retrieve_relevant_context
from app.security.guardrails import validate_search_tool_parameters
from app.tools.errors import ToolTimeoutError
from app.tools.permissions import check_tool_permission


logger = logging.getLogger(__name__)


async def knowledge_search_tool(
    query: str,
    role: str,
    top_k: int = 3,
    metadata_filter: MetadataFilter | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[RetrievedDocument]:
    """Search approved knowledge chunks for the caller's role."""

    metadata = build_run_metadata(
        user_id=user_id,
        role=role,
        session_id=session_id,
        tool_name="knowledge_search_tool",
    )
    with trace_run(
        "knowledge_search_tool",
        run_type="tool",
        inputs={"query": query, "top_k": top_k, "metadata_filter": metadata_filter},
        metadata=metadata,
        tags=["tool", "retrieval"],
    ) as run:
        check_tool_permission("knowledge_search_tool", role)
        validate_search_tool_parameters(query, role, top_k, metadata_filter)
        try:
            results = await asyncio.wait_for(
                retrieve_relevant_context(
                    query=query,
                    role=role,
                    limit=top_k,
                    metadata_filter=metadata_filter,
                    user_id=user_id,
                    session_id=session_id,
                ),
                timeout=get_settings().tool_timeout_seconds,
            )
        except TimeoutError as exc:
            logger.exception(
                "Tool execution timed out",
                extra={
                    "component": "tool",
                    "operation": "knowledge_search_tool",
                    "error_type": type(exc).__name__,
                    "fallback": "graph_safe_response",
                    "user_id": user_id,
                    "role": role,
                    "session_id": session_id,
                    "tool_name": "knowledge_search_tool",
                },
            )
            raise ToolTimeoutError("knowledge_search_tool timed out.") from exc

        set_trace_outputs(
            run,
            {
                "result_count": len(results),
                "citations": [
                    {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "title": result.title,
                        "source_file": result.source_file,
                    }
                    for result in results
                ],
            },
        )
        return results
