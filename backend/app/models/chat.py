"""Chat request and response models.

Pydantic models make the API contract explicit. FastAPI uses these classes to
validate incoming JSON and to document the API schema.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Input accepted by the assistant chat endpoint."""

    user_id: str = Field(..., min_length=1, description="Enterprise user id.")
    role: str = Field(..., min_length=1, description="User role, such as admin or employee.")
    message: str = Field(..., min_length=1, description="User question or task.")
    session_id: str = Field(..., min_length=1, description="Conversation id.")


class AgentActivity(BaseModel):
    """Structured status details for the frontend sidebar."""

    current_state: str
    active_node: str
    tool_calls: list[str] = Field(default_factory=list)
    retrieval_status: str
    validation_results: list[str] = Field(default_factory=list)
    memory_updates: list[str] = Field(default_factory=list)
    activity_log: list[dict[str, str]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """Source information returned with an answer."""

    document_id: str
    chunk_id: str | None = None
    title: str
    snippet: str
    source_file: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """Response returned by the assistant chat endpoint."""

    answer: str
    session_id: str
    agent_activity: AgentActivity
    citations: list[Citation] = Field(default_factory=list)
