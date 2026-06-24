"""Placeholder LangGraph workflow.

LangGraph will eventually coordinate planning, retrieval, tool calls, memory,
and response generation. For now this module keeps the future boundary visible
without adding unnecessary complexity.
"""

from app.memory.store import get_conversation_summary
from app.models.chat import AgentActivity, ChatRequest, ChatResponse, Citation
from app.retrieval.vector_store import retrieve_relevant_context


def build_mock_answer(question: str, citations: list[Citation]) -> str:
    """Build a simple retrieval-grounded answer before real LLM generation exists."""

    if not citations:
        return (
            "I could not find matching local documents for that question. Try asking about "
            "payment incidents, card runbooks, architecture, policies, products, or meetings."
        )

    source_summaries = []
    for citation in citations[:3]:
        source_summaries.append(
            f"- {citation.title}: {citation.snippet.replace(chr(10), ' ')[:300]}"
        )

    return (
        f"Based on the local bank documents, here is what I found for: {question}\n\n"
        + "\n".join(source_summaries)
        + "\n\nThis is still a mock generation step, but the sources are coming from local "
        "hybrid retrieval."
    )


async def run_assistant(request: ChatRequest) -> ChatResponse:
    """Run the assistant workflow.

    This placeholder gives the frontend and API something stable to call while
    the real graph is designed.
    """

    context = await retrieve_relevant_context(request.message, role=request.role)
    memory_summary = get_conversation_summary(request.session_id)
    citations = [
        Citation(
            document_id=document.document_id or document.id,
            chunk_id=document.chunk_id,
            title=document.title,
            snippet=document.snippet,
            source_file=document.source_file,
            score=document.score,
        )
        for document in context
    ]

    return ChatResponse(
        answer=build_mock_answer(request.message, citations),
        session_id=request.session_id,
        agent_activity=AgentActivity(
            current_state="completed_mock_run",
            active_node="final_response",
            tool_calls=["local_hybrid_retriever.search"],
            retrieval_status=f"completed: found {len(context)} placeholder citation(s)",
            validation_results=[
                "request schema valid",
                f"role accepted: {request.role}",
                "rate limit check passed",
            ],
            memory_updates=[
                memory_summary,
                "No persistent memory update was written in the mock backend.",
            ],
        ),
        citations=citations,
    )
