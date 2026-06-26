"""API routes for the assistant."""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agents.graph import run_assistant, stream_assistant_events
from app.models.chat import ChatRequest, ChatResponse
from app.security.guardrails import check_user_role
from app.security.rate_limiter import check_rate_limit


router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.post("/chat/stream", tags=["assistant"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Validate, authorize, and stream assistant activity plus answer tokens."""

    await check_rate_limit(request.user_id)
    check_user_role(request.role)

    async def event_lines():
        try:
            async for event in stream_assistant_events(request):
                yield json.dumps(event) + "\n"
        except Exception as exc:
            logger.exception(
                "Streaming route failed",
                extra={
                    "component": "api",
                    "operation": "chat_stream",
                    "error_type": type(exc).__name__,
                    "fallback": "stream_error_event",
                    "user_id": request.user_id,
                    "role": request.role,
                    "session_id": request.session_id,
                },
            )
            yield json.dumps(
                {
                    "type": "error",
                    "data": {
                        "message": (
                            "I hit a temporary backend issue while streaming that request. "
                            "Please try again in a moment."
                        ),
                        "session_id": request.session_id,
                    },
                }
            ) + "\n"

    return StreamingResponse(event_lines(), media_type="application/x-ndjson")
