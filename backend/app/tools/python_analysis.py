"""Structured incident analytics tool."""

from collections import Counter
from typing import Any

from app.security.guardrails import validate_incident_records
from app.tools.permissions import check_tool_permission


def _read_field(record: dict[str, Any], field_name: str) -> str:
    """Read a normalized string field from structured incident data."""

    value = record.get(field_name) or "unknown"
    return str(value).strip().lower() or "unknown"


def python_analysis_tool(
    incidents: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    """Group incidents by root cause, department, and date."""

    check_tool_permission("python_analysis_tool", role)
    validate_incident_records(incidents)

    root_causes = Counter(_read_field(incident, "root_cause") for incident in incidents)
    departments = Counter(_read_field(incident, "department") for incident in incidents)
    dates = Counter(_read_field(incident, "date") for incident in incidents)

    return {
        "total_incidents": len(incidents),
        "by_root_cause": dict(root_causes),
        "by_department": dict(departments),
        "by_date": dict(dates),
        "top_root_cause": root_causes.most_common(1)[0][0] if root_causes else None,
        "top_department": departments.most_common(1)[0][0] if departments else None,
    }
