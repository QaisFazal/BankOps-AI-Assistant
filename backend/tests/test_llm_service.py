"""Tests for Gemini LLM service fallbacks."""

import asyncio
from types import SimpleNamespace

from app.models.chat import Citation
from app.models.documents import RetrievedDocument
from app.services import llm_service


def test_gemini_missing_api_key_returns_none(monkeypatch) -> None:
    """Missing Gemini credentials should trigger deterministic graph fallback."""

    monkeypatch.setattr(
        llm_service,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_api_key=None,
            gemini_model="gemini-1.5-flash",
            tool_timeout_seconds=5.0,
        ),
    )

    result = asyncio.run(
        llm_service.generate_answer(
            question="Summarize payment incidents.",
            role="analyst",
            user_id="user-1",
            session_id="session-1",
            retrieval_results=[
                RetrievedDocument(
                    id="chunk-1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    title="Payment Incident",
                    snippet="Payment incident context.",
                    score=0.9,
                    source_file="incidents/payment.md",
                    department="payments",
                    document_type="incident",
                    access_level="internal",
                    created_date="2025-01-01",
                )
            ],
            citations=[
                Citation(
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    title="Payment Incident",
                    snippet="Payment incident context.",
                    source_file="incidents/payment.md",
                    score=0.9,
                )
            ],
            conversation_history=[],
            analysis_result="Top source: Payment Incident.",
            errors=[],
        )
    )

    assert result is None
