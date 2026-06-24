"""Basic security placeholders.

Enterprise assistants need careful handling for PII, access control, prompt
injection, and audit trails. These functions are deliberately conservative
stubs for the first milestone.
"""

from fastapi import HTTPException, status


ALLOWED_ROLES = {"administrator", "analyst", "viewer"}


def check_user_role(role: str) -> None:
    """Reject roles that should not call the assistant."""

    if role.lower() not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User role is not allowed to use the assistant.",
        )
