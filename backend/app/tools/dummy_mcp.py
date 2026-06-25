"""Dummy MCP-style enterprise data tool.

This is not a real MCP server. It mimics the shape of an enterprise data tool so
the assistant can be tested against directory, catalog, and incident resources.
"""

from typing import Any, Literal

from app.observability.tracing import build_run_metadata, set_trace_outputs, trace_run
from app.tools.permissions import check_tool_permission


DummyResource = Literal["employee_directory", "service_catalog", "incident_records"]
ALLOWED_RESOURCES = {"employee_directory", "service_catalog", "incident_records"}


EMPLOYEE_DIRECTORY = [
    {
        "employee_id": "E-1001",
        "name": "Maya Fernando",
        "department": "payments",
        "role": "Payments Platform Lead",
    },
    {
        "employee_id": "E-1002",
        "name": "Daniel Perera",
        "department": "cards",
        "role": "Card Operations Manager",
    },
]

SERVICE_CATALOG = [
    {
        "service_id": "svc-payments-gateway",
        "name": "Payment Gateway",
        "owner_department": "payments",
        "tier": "tier_1",
    },
    {
        "service_id": "svc-card-auth",
        "name": "Card Authorization Service",
        "owner_department": "cards",
        "tier": "tier_1",
    },
]

INCIDENT_RECORDS = [
    {
        "incident_id": "INC-2025-001",
        "title": "Payment Gateway Timeout",
        "department": "payments",
        "date": "2025-01-14",
        "root_cause": "vendor certificate latency",
    },
    {
        "incident_id": "INC-2025-002",
        "title": "Card Authorization Latency",
        "department": "cards",
        "date": "2025-02-03",
        "root_cause": "synchronous fraud enrichment",
    },
]


def dummy_mcp_tool(
    resource: DummyResource,
    role: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return dummy enterprise data for authorized roles."""

    metadata = build_run_metadata(
        user_id=user_id,
        role=role,
        session_id=session_id,
        tool_name="dummy_mcp_tool",
        resource=resource,
    )
    with trace_run(
        "dummy_mcp_tool",
        run_type="tool",
        inputs={"resource": resource},
        metadata=metadata,
        tags=["tool", "mcp"],
    ) as run:
        check_tool_permission("dummy_mcp_tool", role)
        if resource not in ALLOWED_RESOURCES:
            raise ValueError(f"Unknown dummy MCP resource: {resource}.")

        if resource == "employee_directory":
            result = EMPLOYEE_DIRECTORY
        elif resource == "service_catalog":
            result = SERVICE_CATALOG
        elif resource == "incident_records":
            result = INCIDENT_RECORDS
        else:
            raise ValueError(f"Unknown dummy MCP resource: {resource}.")

        set_trace_outputs(run, {"record_count": len(result)})
        return result
