"""Tests for session-based conversational memory."""

from app.memory.store import (
    MAX_TURNS_BEFORE_SUMMARY,
    get_active_topic,
    get_conversation_history,
    get_conversation_summary,
    get_session_memory,
    reset_memory_store,
    save_conversation_turn,
)


def setup_function() -> None:
    """Reset global memory before each test."""

    reset_memory_store()


def test_memory_stores_questions_and_answers_by_session() -> None:
    """Each session should retain its own conversation turns."""

    save_conversation_turn("session-a", "Question A?", "Answer A.")
    save_conversation_turn("session-b", "Question B?", "Answer B.")

    history_a = get_conversation_history("session-a")
    history_b = get_conversation_history("session-b")

    assert "User: Question A?" in history_a
    assert "Assistant: Answer A." in history_a
    assert "User: Question B?" not in history_a
    assert "User: Question B?" in history_b


def test_memory_stores_active_topic_and_standalone_question() -> None:
    """Structured conversational fields should survive between turns."""

    save_conversation_turn(
        "session-a",
        "What was its impact?",
        "Customers experienced delays.",
        standalone_question="What was the impact of the payment gateway incident?",
        active_topic="Payment Gateway Timeout Incident",
    )

    memory = get_session_memory("session-a")

    assert get_active_topic("session-a") == "Payment Gateway Timeout Incident"
    assert memory.turns[0].standalone_question == (
        "What was the impact of the payment gateway incident?"
    )


def test_memory_summarizes_when_too_long() -> None:
    """Older turns should compact into summary after the configured limit."""

    updates = []
    for index in range(MAX_TURNS_BEFORE_SUMMARY + 1):
        updates = save_conversation_turn(
            "long-session",
            f"Question {index}?",
            f"Answer {index}.",
        )

    history = get_conversation_history("long-session")
    summary = get_conversation_summary("long-session")

    assert any("Summarized older conversation" in update for update in updates)
    assert history[0].startswith("Summary of earlier conversation:")
    assert "Question 0?" in summary
    assert "Question 4?" in summary
    assert len([item for item in history if item.startswith("User:")]) == 2
