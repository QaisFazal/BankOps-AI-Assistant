"""Security guardrails and role checks.

Enterprise assistants need careful handling for PII, access control, prompt
injection, metadata filtering, tool parameters, and audit trails.
"""

import re
from typing import Any

from fastapi import HTTPException, status


ALLOWED_ROLES = {"administrator", "analyst", "viewer"}
ROLE_ACCESS_LEVELS = {
    "viewer": {"internal"},
    "analyst": {"internal", "confidential"},
    "administrator": {"internal", "confidential", "restricted"},
}

MAX_QUESTION_LENGTH = 2_000
MAX_TOOL_QUERY_LENGTH = 1_000
MAX_TOP_K = 10
ALLOWED_METADATA_FILTERS = {
    "department",
    "document_type",
    "access_level",
    "created_date",
    "source_file",
    "title",
}

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "show hidden system prompt",
    "reveal system prompt",
    "developer message",
    "bypass access",
    "bypass permissions",
    "disable guardrails",
    "export all confidential documents",
    "exfiltrate",
    "show hidden tools",
)

PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    r"\b(show|reveal|print|dump)\s+(me\s+)?(the\s+)?(hidden\s+)?system\s+prompt\b",
    r"\bbypass\s+(access|permissions|authorization|rbac)\b",
    r"\b(disable|turn\s+off)\s+guardrails\b",
    r"\b(export|show|dump|list)\s+(all\s+)?"
    r"(confidential|restricted|admin)(\s+(confidential|restricted|admin))*\s+"
    r"(documents|docs|files|data)\b",
)


class GuardrailViolation(ValueError):
    """Raised when a request or tool call violates a security guardrail."""


def check_user_role(role: str) -> None:
    """Reject roles that should not call the assistant."""

    if role.lower() not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User role is not allowed to use the assistant.",
        )


def detect_prompt_injection(text: str) -> list[str]:
    """Return prompt-injection markers found in user-controlled text."""

    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    detected_markers = [marker for marker in PROMPT_INJECTION_MARKERS if marker in lowered]
    detected_markers.extend(
        pattern
        for pattern in PROMPT_INJECTION_PATTERNS
        if re.search(pattern, lowered)
    )
    return detected_markers


def validate_user_question(question: str) -> None:
    """Validate user question length and basic shape."""

    if not question.strip():
        raise GuardrailViolation("Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise GuardrailViolation(
            f"Question is too long. Maximum length is {MAX_QUESTION_LENGTH} characters."
        )


def normalize_filter_values(value: str | list[str]) -> list[str]:
    """Normalize metadata filter values for authorization checks."""

    values = value if isinstance(value, list) else [value]
    return [str(item).strip().lower() for item in values if str(item).strip()]


def validate_metadata_filter(role: str, metadata_filter: dict[str, Any] | None) -> None:
    """Validate metadata filters and block access-level escalation."""

    if not metadata_filter:
        return

    unknown_fields = set(metadata_filter).difference(ALLOWED_METADATA_FILTERS)
    if unknown_fields:
        raise GuardrailViolation(
            "Unsupported metadata filter(s): " + ", ".join(sorted(unknown_fields))
        )

    access_filter = metadata_filter.get("access_level")
    if access_filter is None:
        return

    requested_access_levels = set(normalize_filter_values(access_filter))
    allowed_access_levels = ROLE_ACCESS_LEVELS.get(role.lower(), set())
    blocked_levels = requested_access_levels.difference(allowed_access_levels)
    if blocked_levels:
        raise GuardrailViolation(
            f"Role {role} cannot filter for access level(s): {', '.join(sorted(blocked_levels))}."
        )


def validate_search_tool_parameters(
    query: str,
    role: str,
    top_k: int,
    metadata_filter: dict[str, Any] | None,
) -> None:
    """Validate common knowledge-search tool parameters."""

    if not query.strip():
        raise GuardrailViolation("Search query cannot be empty.")
    if len(query) > MAX_TOOL_QUERY_LENGTH:
        raise GuardrailViolation(
            f"Search query is too long. Maximum length is {MAX_TOOL_QUERY_LENGTH} characters."
        )
    if top_k < 1 or top_k > MAX_TOP_K:
        raise GuardrailViolation(f"top_k must be between 1 and {MAX_TOP_K}.")

    validate_metadata_filter(role, metadata_filter)


def validate_incident_records(incidents: list[dict[str, Any]]) -> None:
    """Validate structured incident records for analytics."""

    required_fields = {"root_cause", "department", "date"}
    if len(incidents) > 500:
        raise GuardrailViolation("Too many incidents supplied for local analysis.")

    for index, incident in enumerate(incidents):
        missing_fields = required_fields.difference(incident)
        if missing_fields:
            raise GuardrailViolation(
                f"Incident {index} missing field(s): {', '.join(sorted(missing_fields))}."
            )
