"""Placeholder LangGraph workflow.

LangGraph will eventually coordinate planning, retrieval, tool calls, memory,
and response generation. For now this module keeps the future boundary visible
without adding unnecessary complexity.
"""

from app.memory.store import get_conversation_summary
from app.models.chat import ChatRequest, ChatResponse
from app.retrieval.vector_store import retrieve_relevant_context


def run_assistant(request: ChatRequest) -> ChatResponse:
    """Run the assistant workflow.

    This placeholder gives the frontend and API something stable to call while
    the real graph is designed.
    """

    context = retrieve_relevant_context(request.message)
    memory_summary = get_conversation_summary(request.session_id)

    return ChatResponse(
        answer=(
            "This is a scaffolded response. Next, connect LangGraph nodes for "
            "intent routing, retrieval, tool use, and final answer generation."
        ),
        session_id=request.session_id,
        sources=context,
        debug_notes=[
            f"Loaded memory summary: {memory_summary}",
            "LangGraph workflow is not implemented yet.",
        ],
    )
