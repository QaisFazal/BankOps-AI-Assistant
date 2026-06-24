"""Chat request and response models."""

from pydantic import BaseModel, Field

from app.models.documents import RetrievedDocument


class ChatRequest(BaseModel):
    """Input accepted by the assistant chat endpoint."""

    message: str = Field(..., min_length=1, description="User question or task.")
    session_id: str = Field(default="demo-session", description="Conversation id.")
    user_id: str | None = Field(default=None, description="Optional enterprise user id.")


class ChatResponse(BaseModel):
    """Response returned by the assistant chat endpoint."""

    answer: str
    session_id: str
    sources: list[RetrievedDocument] = Field(default_factory=list)
    debug_notes: list[str] = Field(default_factory=list)
