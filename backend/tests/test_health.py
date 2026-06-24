"""Smoke tests for the FastAPI app."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    """The app should expose a simple health endpoint."""

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
