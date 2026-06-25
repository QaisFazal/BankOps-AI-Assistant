"""Central registry for assistant tools."""


def list_tools() -> list[dict[str, str]]:
    """Return metadata for tools available to the assistant."""

    return [
        {
            "name": "knowledge_search_tool",
            "description": "Search approved enterprise knowledge with hybrid retrieval.",
        },
        {
            "name": "python_analysis_tool",
            "description": "Analyze structured incident data for root cause, department, and date.",
        },
        {
            "name": "dummy_mcp_tool",
            "description": "Read dummy enterprise directory, service catalog, or incident records.",
        },
    ]
