"""Tests for assistant tool permissions and behavior."""

import asyncio

import pytest

from app.tools.dummy_mcp import dummy_mcp_tool
from app.tools.knowledge_search import knowledge_search_tool
from app.tools.permissions import ToolPermissionError, can_execute_tool
from app.tools.python_analysis import python_analysis_tool
from app.security.guardrails import GuardrailViolation


def test_viewer_cannot_execute_python_analysis_tool() -> None:
    """Viewer role should not be allowed to run analytics."""

    incidents = [
        {
            "root_cause": "certificate latency",
            "department": "payments",
            "date": "2025-01-14",
        }
    ]

    with pytest.raises(ToolPermissionError):
        python_analysis_tool(incidents, role="viewer")


def test_viewer_cannot_execute_dummy_mcp_tool() -> None:
    """Viewer role should not be allowed to read dummy enterprise MCP data."""

    with pytest.raises(ToolPermissionError):
        dummy_mcp_tool("employee_directory", role="viewer")


def test_analyst_can_group_incidents() -> None:
    """Analysts can group incident data by root cause, department, and date."""

    result = python_analysis_tool(
        [
            {
                "root_cause": "certificate latency",
                "department": "payments",
                "date": "2025-01-14",
            },
            {
                "root_cause": "certificate latency",
                "department": "payments",
                "date": "2025-01-14",
            },
            {
                "root_cause": "fraud enrichment",
                "department": "cards",
                "date": "2025-02-03",
            },
        ],
        role="analyst",
    )

    assert result["total_incidents"] == 3
    assert result["by_root_cause"]["certificate latency"] == 2
    assert result["by_department"]["payments"] == 2
    assert result["by_date"]["2025-01-14"] == 2


def test_administrator_can_execute_dummy_mcp_tool() -> None:
    """Administrators can read dummy enterprise resources."""

    result = dummy_mcp_tool("service_catalog", role="administrator")

    assert result[0]["service_id"] == "svc-payments-gateway"


def test_viewer_can_execute_knowledge_search_tool() -> None:
    """Knowledge search is allowed for viewer with retrieval metadata filtering."""

    results = asyncio.run(
        knowledge_search_tool(
            "card authorization latency",
            role="viewer",
            top_k=1,
        )
    )

    assert isinstance(results, list)


def test_hardcoded_rbac_matrix() -> None:
    """The hardcoded matrix should match the product role requirements."""

    assert can_execute_tool("knowledge_search_tool", "viewer")
    assert not can_execute_tool("python_analysis_tool", "viewer")
    assert not can_execute_tool("dummy_mcp_tool", "viewer")

    assert can_execute_tool("knowledge_search_tool", "analyst")
    assert can_execute_tool("python_analysis_tool", "analyst")
    assert can_execute_tool("dummy_mcp_tool", "analyst")

    assert can_execute_tool("knowledge_search_tool", "administrator")
    assert can_execute_tool("python_analysis_tool", "administrator")
    assert can_execute_tool("dummy_mcp_tool", "administrator")


def test_viewer_cannot_search_confidential_metadata_filter() -> None:
    """Viewer should not be allowed to request confidential search filters."""

    with pytest.raises(GuardrailViolation):
        asyncio.run(
            knowledge_search_tool(
                "payment policy",
                role="viewer",
                metadata_filter={"access_level": "confidential"},
            )
        )


def test_search_tool_rejects_invalid_top_k() -> None:
    """Search tool should validate top_k before retrieval."""

    with pytest.raises(GuardrailViolation):
        asyncio.run(knowledge_search_tool("payment policy", role="analyst", top_k=0))


def test_search_tool_rejects_unknown_metadata_filter() -> None:
    """Search tool should reject unsupported metadata filter keys."""

    with pytest.raises(GuardrailViolation):
        asyncio.run(
            knowledge_search_tool(
                "payment policy",
                role="analyst",
                metadata_filter={"secret_override": "true"},
            )
        )


def test_python_analysis_tool_rejects_malformed_incident_records() -> None:
    """Analytics tool should validate required incident fields."""

    with pytest.raises(GuardrailViolation):
        python_analysis_tool([{"department": "payments"}], role="analyst")
