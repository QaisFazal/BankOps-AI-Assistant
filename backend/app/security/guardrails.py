"""Basic security placeholders.

Enterprise assistants need careful handling for PII, access control, prompt
injection, and audit trails. These functions are deliberately conservative
stubs for the first milestone.
"""


def is_request_allowed(user_id: str | None) -> bool:
    """Return whether a request should be processed."""

    _ = user_id
    return True
