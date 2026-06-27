"""In-memory conversational memory store.

This is intentionally process-local and simple. It is useful for local demos and
tests, but a production assistant should move this state to durable storage.
"""

from dataclasses import dataclass, field
from threading import Lock


MAX_TURNS_BEFORE_SUMMARY = 4
RECENT_TURNS_TO_KEEP = 2
ANSWER_PREVIEW_CHARS = 180


@dataclass
class ConversationTurn:
    """One user/assistant exchange."""

    question: str
    answer: str
    standalone_question: str | None = None
    active_topic: str | None = None


@dataclass
class SessionMemory:
    """Memory tracked for one conversation session."""

    summary: str | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    active_topic: str | None = None


_memory_by_session: dict[str, SessionMemory] = {}
_memory_lock = Lock()


def _preview(text: str, limit: int = ANSWER_PREVIEW_CHARS) -> str:
    """Return a compact single-line preview for memory summaries."""

    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact

    return compact[: limit - 3] + "..."


def _summarize_memory(memory: SessionMemory) -> str:
    """Create a simple deterministic summary for older conversation turns."""

    previous_summary = memory.summary or "No prior summary."
    older_turns = memory.turns[:-RECENT_TURNS_TO_KEEP]
    turn_summaries = [
        f"User asked '{_preview(turn.question, 90)}'; assistant answered '{_preview(turn.answer, 120)}'."
        for turn in older_turns
    ]

    return " ".join([previous_summary, *turn_summaries]).strip()


def get_session_memory(session_id: str) -> SessionMemory:
    """Return the mutable memory object for a session."""

    with _memory_lock:
        return _memory_by_session.setdefault(session_id, SessionMemory())


def get_conversation_history(session_id: str) -> list[str]:
    """Return summary plus recent turns as readable history strings."""

    with _memory_lock:
        memory = _memory_by_session.get(session_id)
        if memory is None:
            return []

        history: list[str] = []
        if memory.summary:
            history.append(f"Summary of earlier conversation: {memory.summary}")

        for turn in memory.turns:
            history.append(f"User: {turn.question}")
            history.append(f"Assistant: {_preview(turn.answer)}")

        return history


def get_conversation_summary(session_id: str) -> str:
    """Return a short memory summary for a session."""

    history = get_conversation_history(session_id)
    if not history:
        return f"No stored memory yet for session {session_id}."

    return " | ".join(history)


def get_active_topic(session_id: str) -> str | None:
    """Return the latest structured topic for contextual retrieval."""

    with _memory_lock:
        memory = _memory_by_session.get(session_id)
        return memory.active_topic if memory is not None else None


def save_conversation_turn(
    session_id: str,
    question: str,
    answer: str,
    *,
    standalone_question: str | None = None,
    active_topic: str | None = None,
) -> list[str]:
    """Store one turn and summarize older memory when it grows too long."""

    updates = [f"Stored latest turn for session {session_id}."]
    with _memory_lock:
        memory = _memory_by_session.setdefault(session_id, SessionMemory())
        memory.turns.append(
            ConversationTurn(
                question=question,
                answer=answer,
                standalone_question=standalone_question,
                active_topic=active_topic,
            )
        )
        if active_topic:
            memory.active_topic = active_topic

        if len(memory.turns) > MAX_TURNS_BEFORE_SUMMARY:
            memory.summary = _summarize_memory(memory)
            memory.turns = memory.turns[-RECENT_TURNS_TO_KEEP:]
            updates.append(
                f"Summarized older conversation; kept {len(memory.turns)} recent turn(s)."
            )

    return updates


def reset_memory_store() -> None:
    """Clear memory. Intended for tests and local development resets."""

    with _memory_lock:
        _memory_by_session.clear()
