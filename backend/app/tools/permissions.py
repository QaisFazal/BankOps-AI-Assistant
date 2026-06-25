"""Role checks shared by assistant tools."""

from app.security.guardrails import ALLOWED_ROLES


class ToolPermissionError(PermissionError):
    """Raised when a role is not allowed to execute a tool."""


TOOL_ALLOWED_ROLES = {
    "knowledge_search_tool": {"viewer", "analyst", "administrator"},
    "python_analysis_tool": {"analyst", "administrator"},
    "dummy_mcp_tool": {"analyst", "administrator"},
}


def can_execute_tool(tool_name: str, role: str) -> bool:
    """Return whether the role may execute a tool."""

    normalized_role = role.lower()
    return normalized_role in TOOL_ALLOWED_ROLES.get(tool_name, set())


def check_tool_permission(tool_name: str, role: str) -> None:
    """Ensure the user's role may execute a tool."""

    normalized_role = role.lower()
    if normalized_role not in ALLOWED_ROLES:
        raise ToolPermissionError(f"Unknown role cannot execute {tool_name}: {role}.")

    if not can_execute_tool(tool_name, normalized_role):
        raise ToolPermissionError(f"Role {role} cannot execute {tool_name}.")
