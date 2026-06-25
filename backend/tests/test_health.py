"""API smoke tests for the FastAPI app."""

from fastapi.testclient import TestClient

from app.main import app
from app.memory.store import reset_memory_store
from app.security.rate_limiter import reset_rate_limiter


client = TestClient(app)


def setup_function() -> None:
    """Reset shared state before API tests."""

    reset_memory_store()
    reset_rate_limiter()


def test_health_check() -> None:
    """The app should expose a simple health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_returns_mock_answer() -> None:
    """The chat endpoint should return the mocked backend flow output."""

    response = client.post(
        "/chat",
        json={
            "user_id": "user-123",
            "role": "analyst",
            "message": "Summarize the AI policy.",
            "session_id": "session-123",
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert data["session_id"] == "session-123"
    assert data["answer"].startswith("I could not find matching local documents") or data[
        "answer"
    ].startswith("Here is a grounded summary")
    assert data["agent_activity"]["current_state"] == "completed_langgraph_run"
    assert data["agent_activity"]["active_node"] == "citation_validation_node"
    assert any(
        tool_call.startswith("knowledge_search_tool")
        for tool_call in data["agent_activity"]["tool_calls"]
    )
    assert data["agent_activity"]["activity_log"][0]["node"] == "memory"
    assert "citations" in data


def test_chat_requires_expected_fields() -> None:
    """FastAPI and Pydantic should reject incomplete chat requests."""

    response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 422


def test_chat_rejects_unsupported_role() -> None:
    """The role guard should block roles outside the starter allowlist."""

    response = client.post(
        "/chat",
        json={
            "user_id": "user-456",
            "role": "guest",
            "message": "Can I use the assistant?",
            "session_id": "session-456",
        },
    )

    assert response.status_code == 403


def test_chat_returns_graceful_429_when_rate_limited() -> None:
    """The chat endpoint should return structured rate-limit activity."""

    reset_rate_limiter(capacity=1, refill_rate_per_second=0)
    payload = {
        "user_id": "limited-user",
        "role": "analyst",
        "message": "How do we handle card authorization latency?",
        "session_id": "limited-session",
    }

    first_response = client.post("/chat", json=payload)
    second_response = client.post("/chat", json=payload)

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    data = second_response.json()
    assert data["detail"] == "Rate limit exceeded. Please try again shortly."
    assert data["activity_log"][0]["node"] == "rate_limiter"
    assert data["activity_log"][0]["status"] == "blocked"
