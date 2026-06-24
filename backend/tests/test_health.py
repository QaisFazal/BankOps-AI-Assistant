"""API smoke tests for the FastAPI app."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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
    ].startswith("Based on the local bank documents")
    assert data["agent_activity"]["current_state"] == "completed_mock_run"
    assert data["agent_activity"]["active_node"] == "final_response"
    assert data["agent_activity"]["tool_calls"] == ["local_hybrid_retriever.search"]
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
