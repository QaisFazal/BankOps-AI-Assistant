"""API routes for the assistant.

The current endpoint returns a mock response while the LangGraph workflow,
retrieval, and tools are still placeholders.
"""

from fastapi import APIRouter

from app.agents.graph import run_assistant
from app.models.chat import ChatRequest, ChatResponse


router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["assistant"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Accept a user question and return a starter assistant response."""

    return run_assistant(request)
