"""API routes for the assistant."""

from fastapi import APIRouter

from app.agents.graph import run_assistant
from app.models.chat import ChatRequest, ChatResponse
from app.security.guardrails import check_user_role
from app.security.rate_limiter import check_rate_limit


router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["assistant"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Validate, authorize, and process a chat request.

    Flow:
    validate request -> check rate limit -> check role -> run mock LangGraph agent
    -> return answer, agent activity, and citations.
    """

    await check_rate_limit(request.user_id)
    check_user_role(request.role)
    return await run_assistant(request)
