"""Placeholder memory store.

Start with simple functions before introducing a database. Later, this package
can hold conversation summaries, preferences, and per-session state.
"""


def get_conversation_summary(session_id: str) -> str:
    """Return a short summary for the current conversation session."""

    return f"No stored memory yet for session {session_id}."
