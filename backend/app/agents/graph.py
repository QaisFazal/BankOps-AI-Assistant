"""LangGraph workflow for the assistant.

The graph is still model-free: it uses rules and retrieved chunks to produce a
simple grounded answer. The important part for this milestone is that each step
has a clear node, state update, and activity log entry.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.memory.store import get_conversation_history, save_conversation_turn
from app.models.chat import AgentActivity, ChatRequest, ChatResponse, Citation
from app.models.documents import RetrievedDocument
from app.retrieval.base import MetadataFilter
from app.tools.permissions import ToolPermissionError, can_execute_tool
from app.tools.knowledge_search import knowledge_search_tool


class ActivityLogEntry(TypedDict):
    """One node-level event for the UI and debugging."""

    node: str
    status: str
    message: str


class SearchPlan(TypedDict):
    """Simple recursive search plan for broad questions."""

    objective: str
    sub_queries: list[str]
    filters: MetadataFilter
    batch_strategy: str
    aggregation_strategy: str


class BatchAnalysis(TypedDict):
    """Summary of one retrieval batch."""

    depth: int
    query: str
    result_count: int
    summary: str


class AssistantState(TypedDict):
    """State carried through the LangGraph assistant workflow."""

    user_id: str
    role: str
    session_id: str
    question: str
    conversation_history: list[str]
    intent: str
    search_plan: SearchPlan | None
    max_depth: int
    retrieval_results: list[RetrievedDocument]
    tool_calls: list[str]
    batch_analyses: list[BatchAnalysis]
    analysis_result: str
    validation_results: list[str]
    final_answer: str
    memory_updates: list[str]
    activity_log: list[ActivityLogEntry]
    errors: list[str]
    citations: list[Citation]


PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "reveal system prompt",
    "developer message",
    "bypass access",
    "disable guardrails",
    "exfiltrate",
    "show hidden tools",
)

BROAD_QUESTION_MARKERS = (
    "overview",
    "summarize",
    "summary",
    "compare",
    "all",
    "across",
    "everything",
    "what should i know",
    "tell me about",
    "landscape",
    "strategy",
)

ANALYTICS_REQUEST_MARKERS = (
    "analytics",
    "analysis",
    "group incidents",
    "group by",
    "root cause",
    "summary statistics",
    "incident statistics",
    "trend",
)

MCP_REQUEST_MARKERS = (
    "employee directory",
    "service catalog",
    "incident records",
    "mcp",
    "enterprise data",
)


def append_activity(
    state: AssistantState,
    node: str,
    status: str,
    message: str,
) -> AssistantState:
    """Append a node activity event to state."""

    state["activity_log"].append({"node": node, "status": status, "message": message})
    return state


def classify_intent(question: str) -> str:
    """Classify the user's question into a simple routing intent."""

    lowered = question.lower()
    if any(term in lowered for term in ("incident", "outage", "latency", "delay")):
        return "incident_support"
    if any(term in lowered for term in ("runbook", "steps", "how do we handle")):
        return "runbook_lookup"
    if any(term in lowered for term in ("policy", "security", "retention", "handling")):
        return "policy_lookup"
    if any(term in lowered for term in ("architecture", "platform", "design")):
        return "architecture_lookup"
    if any(term in lowered for term in ("meeting", "decision", "notes")):
        return "meeting_summary"

    return "general_knowledge_lookup"


def requested_restricted_tool(question: str) -> str | None:
    """Infer whether the question asks for a restricted tool."""

    lowered = question.lower()
    if any(marker in lowered for marker in ANALYTICS_REQUEST_MARKERS):
        return "python_analysis_tool"
    if any(marker in lowered for marker in MCP_REQUEST_MARKERS):
        return "dummy_mcp_tool"

    return None


def is_broad_question(question: str) -> bool:
    """Return whether the question should be decomposed before retrieval."""

    lowered = question.lower()
    return any(marker in lowered for marker in BROAD_QUESTION_MARKERS) or len(lowered.split()) > 14


def generate_search_plan(question: str, intent: str) -> SearchPlan:
    """Generate a small deterministic search plan for broad questions."""

    filters: MetadataFilter = {}
    if intent == "incident_support":
        filters = {"document_type": "incident"}
        sub_queries = [
            f"{question} payment outage incident impact",
            f"{question} card authorization latency incident",
            f"{question} settlement or bill pay incident follow-up",
        ]
    elif intent == "runbook_lookup":
        filters = {"document_type": "runbook"}
        sub_queries = [
            f"{question} card processing runbook steps",
            f"{question} payment queue runbook steps",
            f"{question} degraded mode or failover runbook",
        ]
    elif intent == "policy_lookup":
        filters = {"document_type": "policy"}
        sub_queries = [
            f"{question} AI assistant security policy",
            f"{question} payment data retention policy",
            f"{question} cardholder data handling policy",
        ]
    elif intent == "architecture_lookup":
        filters = {"document_type": "architecture"}
        sub_queries = [
            f"{question} real-time payments architecture",
            f"{question} card processing architecture",
            f"{question} AI assistant retrieval architecture",
        ]
    else:
        sub_queries = [
            f"{question} incidents",
            f"{question} runbooks",
            f"{question} policies and architecture",
        ]

    return {
        "objective": f"Answer the broad question with evidence from multiple relevant document batches: {question}",
        "sub_queries": sub_queries,
        "filters": filters,
        "batch_strategy": "retrieve each sub-query independently with role-based metadata filtering",
        "aggregation_strategy": "deduplicate chunks, summarize each batch, then combine strongest evidence",
    }


def summarize_batch(query: str, depth: int, results: list[RetrievedDocument]) -> BatchAnalysis:
    """Create a short analysis entry for one recursive retrieval batch."""

    if not results:
        summary = "No matching chunks found."
    else:
        titles = "; ".join(result.title for result in results[:3])
        summary = f"Top source(s): {titles}."

    return {
        "depth": depth,
        "query": query,
        "result_count": len(results),
        "summary": summary,
    }


async def execute_recursive_search(
    state: AssistantState,
    query: str,
    depth: int,
    metadata_filter: MetadataFilter | None,
) -> list[RetrievedDocument]:
    """Execute retrieval recursively with a hard depth limit."""

    append_activity(
        state,
        "recursive_retrieval",
        "started",
        f"Depth {depth}: retrieving sub-query: {query}",
    )
    state["tool_calls"].append(f"knowledge_search_tool(depth={depth})")
    try:
        results = await knowledge_search_tool(
            query,
            role=state["role"],
            top_k=3,
            metadata_filter=metadata_filter,
        )
    except ToolPermissionError as exc:
        state["errors"].append(f"Unauthorized tool call blocked: {exc}")
        state["validation_results"].append("tool authorization failed")
        append_activity(
            state,
            "recursive_retrieval",
            "blocked",
            str(exc),
        )
        return []
    state["batch_analyses"].append(summarize_batch(query, depth, results))
    append_activity(
        state,
        "recursive_retrieval",
        "completed",
        f"Depth {depth}: retrieved {len(results)} chunk(s).",
    )

    if depth >= state["max_depth"] or not is_broad_question(query):
        return results

    child_plan = generate_search_plan(query, state["intent"])
    append_activity(
        state,
        "planning_node",
        "recursive_plan",
        f"Depth {depth}: expanded into {len(child_plan['sub_queries'])} child sub-query/sub-queries.",
    )

    child_results = list(results)
    for child_query in child_plan["sub_queries"][:2]:
        child_results.extend(
            await execute_recursive_search(
                state,
                child_query,
                depth + 1,
                child_plan["filters"] or metadata_filter,
            )
        )

    return child_results


def deduplicate_results(results: list[RetrievedDocument]) -> list[RetrievedDocument]:
    """Deduplicate retrieval results by chunk id while preserving rank order."""

    seen: set[str] = set()
    deduplicated: list[RetrievedDocument] = []
    for result in results:
        result_id = result.chunk_id or result.id
        if result_id in seen:
            continue
        seen.add(result_id)
        deduplicated.append(result)

    return deduplicated


async def supervisor_node(state: AssistantState) -> AssistantState:
    """Classify intent and decide that the request should enter the safety path."""

    state["intent"] = classify_intent(state["question"])
    history_count = len(state["conversation_history"])
    return append_activity(
        state,
        "supervisor_node",
        "completed",
        (
            f"Classified intent as {state['intent']} and routed to security_node. "
            f"Loaded {history_count} memory item(s)."
        ),
    )


async def security_node(state: AssistantState) -> AssistantState:
    """Detect prompt injection and tool abuse attempts."""

    question = state["question"].lower()
    suspicious_markers = [marker for marker in PROMPT_INJECTION_MARKERS if marker in question]

    if suspicious_markers:
        state["errors"].append(
            "Potential prompt injection or tool abuse detected: "
            + ", ".join(suspicious_markers)
        )
        state["validation_results"].append("security check failed")
        return append_activity(
            state,
            "security_node",
            "blocked",
            "Blocked request before retrieval because suspicious instruction markers were found.",
        )

    restricted_tool = requested_restricted_tool(state["question"])
    if restricted_tool and not can_execute_tool(restricted_tool, state["role"]):
        state["errors"].append(
            f"Unauthorized: role {state['role']} cannot execute {restricted_tool}."
        )
        state["validation_results"].append("tool authorization failed")
        return append_activity(
            state,
            "security_node",
            "blocked",
            f"Blocked unauthorized request for {restricted_tool}.",
        )

    state["validation_results"].append("security check passed")
    return append_activity(
        state,
        "security_node",
        "completed",
        "No prompt injection or tool abuse markers detected.",
    )


def route_after_security(state: AssistantState) -> Literal["planning_node", "response_node"]:
    """Skip retrieval when security has already blocked the request."""

    if state["errors"]:
        return "response_node"

    return "planning_node"


async def planning_node(state: AssistantState) -> AssistantState:
    """Generate a bounded recursive search plan for broad questions."""

    if not is_broad_question(state["question"]):
        state["search_plan"] = None
        return append_activity(
            state,
            "planning_node",
            "skipped",
            "Question is narrow; using direct retrieval.",
        )

    state["search_plan"] = generate_search_plan(state["question"], state["intent"])
    plan = state["search_plan"]
    return append_activity(
        state,
        "planning_node",
        "completed",
        (
            f"Generated SearchPlan with objective '{plan['objective']}', "
            f"{len(plan['sub_queries'])} sub-query/sub-queries, "
            f"batch_strategy='{plan['batch_strategy']}', "
            f"aggregation_strategy='{plan['aggregation_strategy']}'."
        ),
    )


async def retrieval_node(state: AssistantState) -> AssistantState:
    """Call the configured hybrid retriever."""

    if state["search_plan"] is None:
        state["tool_calls"].append("knowledge_search_tool")
        try:
            state["retrieval_results"] = await knowledge_search_tool(
                state["question"],
                role=state["role"],
                top_k=3,
            )
        except ToolPermissionError as exc:
            state["errors"].append(f"Unauthorized tool call blocked: {exc}")
            state["validation_results"].append("tool authorization failed")
            state["retrieval_results"] = []
            return append_activity(
                state,
                "retrieval_node",
                "blocked",
                str(exc),
            )
        state["batch_analyses"].append(
            summarize_batch(state["question"], depth=0, results=state["retrieval_results"])
        )
    else:
        plan = state["search_plan"]
        planned_results: list[RetrievedDocument] = []
        append_activity(
            state,
            "retrieval_node",
            "started",
            f"Executing SearchPlan objective: {plan['objective']}",
        )
        for sub_query in plan["sub_queries"]:
            planned_results.extend(
                await execute_recursive_search(
                    state,
                    sub_query,
                    depth=1,
                    metadata_filter=plan["filters"],
                )
            )
        state["retrieval_results"] = deduplicate_results(planned_results)[:6]

    return append_activity(
        state,
        "retrieval_node",
        "completed",
        f"Retrieved {len(state['retrieval_results'])} chunk(s).",
    )


async def analysis_node(state: AssistantState) -> AssistantState:
    """Analyze retrieved chunks before answer generation."""

    if not state["retrieval_results"]:
        state["analysis_result"] = "No relevant chunks were retrieved."
        return append_activity(
            state,
            "analysis_node",
            "completed",
            "No retrieved chunks were available to analyze.",
        )

    if state["search_plan"] is not None:
        batch_summaries = [
            f"depth {batch['depth']} '{batch['query']}': {batch['summary']}"
            for batch in state["batch_analyses"]
        ]
        state["analysis_result"] = "Aggregated planned search batches. " + " | ".join(
            batch_summaries
        )
        return append_activity(
            state,
            "analysis_node",
            "completed",
            f"Analyzed and aggregated {len(state['batch_analyses'])} recursive retrieval batch(es).",
        )

    source_titles = [result.title for result in state["retrieval_results"][:3]]
    state["analysis_result"] = f"Top sources for intent {state['intent']}: " + "; ".join(
        source_titles
    )
    return append_activity(
        state,
        "analysis_node",
        "completed",
        f"Analyzed top source(s): {', '.join(source_titles)}.",
    )


def build_answer(
    question: str,
    citations: list[Citation],
    errors: list[str],
    analysis_result: str,
    search_plan: SearchPlan | None,
    conversation_history: list[str],
) -> str:
    """Build a simple retrieval-grounded final answer."""

    if errors:
        if any(error.startswith("Unauthorized:") or "Unauthorized tool call" in error for error in errors):
            return (
                "You are not authorized to use the requested tool. "
                "Viewers can chat and search approved knowledge only. "
                "Please ask a document-search question or switch to an authorized role."
            )

        return (
            "I cannot answer that request because it appears to contain unsafe instructions. "
            "Please ask a normal question about the approved enterprise documents."
        )

    memory_note = ""
    if conversation_history:
        memory_note = (
            "\n\nConversation memory used: "
            + " | ".join(conversation_history[-4:])
        )

    if not citations:
        return (
            "I could not find matching local documents for that question. Try asking about "
            "payment incidents, card runbooks, architecture, policies, products, or meetings."
            + memory_note
        )

    source_summaries = [
        f"- {citation.title}: {citation.snippet.replace(chr(10), ' ')[:300]}"
        for citation in citations[:3]
    ]
    planning_note = ""
    if search_plan is not None:
        planning_note = (
            f"\n\nSearch objective: {search_plan['objective']}\n"
            f"Aggregation: {search_plan['aggregation_strategy']}"
        )
    return (
        f"Based on the local bank documents, here is what I found for: {question}\n\n"
        + "\n".join(source_summaries)
        + f"\n\nAnalysis: {analysis_result}"
        + planning_note
        + memory_note
        + "\n\nCitations are attached to this response and validated against retrieved chunks."
    )


async def response_node(state: AssistantState) -> AssistantState:
    """Generate the final answer with citations."""

    state["citations"] = [
        Citation(
            document_id=result.document_id or result.id,
            chunk_id=result.chunk_id,
            title=result.title,
            snippet=result.snippet,
            source_file=result.source_file,
            score=result.score,
        )
        for result in state["retrieval_results"]
    ]
    state["final_answer"] = build_answer(
        state["question"],
        state["citations"],
        state["errors"],
        state["analysis_result"],
        state["search_plan"],
        state["conversation_history"],
    )

    return append_activity(
        state,
        "response_node",
        "completed",
        f"Generated answer with {len(state['citations'])} citation(s).",
    )


async def citation_validation_node(state: AssistantState) -> AssistantState:
    """Ensure every citation maps to one of the retrieved chunks."""

    retrieved_chunk_ids = {
        result.chunk_id
        for result in state["retrieval_results"]
        if result.chunk_id is not None
    }
    invalid_citations = [
        citation.chunk_id
        for citation in state["citations"]
        if citation.chunk_id not in retrieved_chunk_ids
    ]

    if invalid_citations:
        state["errors"].append(
            "Citation validation failed for chunk(s): "
            + ", ".join(str(chunk_id) for chunk_id in invalid_citations)
        )
        state["validation_results"].append("citation validation failed")
        return append_activity(
            state,
            "citation_validation_node",
            "failed",
            "One or more citations did not map to retrieved chunks.",
        )

    state["validation_results"].append("citation validation passed")
    return append_activity(
        state,
        "citation_validation_node",
        "completed",
        "All citations map to retrieved chunks.",
    )


def persist_memory_after_run(state: AssistantState) -> AssistantState:
    """Save the completed user/assistant turn to session memory."""

    updates = save_conversation_turn(
        session_id=state["session_id"],
        question=state["question"],
        answer=state["final_answer"],
    )
    state["memory_updates"].extend(updates)
    state["conversation_history"] = get_conversation_history(state["session_id"])

    for update in updates:
        append_activity(state, "memory_node", "updated", update)

    return state


def build_assistant_graph():
    """Build and compile the assistant LangGraph workflow."""

    graph = StateGraph(AssistantState)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("security_node", security_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("response_node", response_node)
    graph.add_node("citation_validation_node", citation_validation_node)

    graph.add_edge(START, "supervisor_node")
    graph.add_edge("supervisor_node", "security_node")
    graph.add_conditional_edges(
        "security_node",
        route_after_security,
        {
            "planning_node": "planning_node",
            "response_node": "response_node",
        },
    )
    graph.add_edge("planning_node", "retrieval_node")
    graph.add_edge("retrieval_node", "analysis_node")
    graph.add_edge("analysis_node", "response_node")
    graph.add_edge("response_node", "citation_validation_node")
    graph.add_edge("citation_validation_node", END)

    return graph.compile()


assistant_graph = build_assistant_graph()


def initial_state_from_request(request: ChatRequest) -> AssistantState:
    """Create LangGraph state from an API chat request."""

    conversation_history = get_conversation_history(request.session_id)
    memory_message = (
        f"Loaded {len(conversation_history)} memory item(s) for session {request.session_id}."
        if conversation_history
        else f"No stored memory yet for session {request.session_id}."
    )
    return {
        "user_id": request.user_id,
        "role": request.role,
        "session_id": request.session_id,
        "question": request.message,
        "conversation_history": conversation_history,
        "intent": "",
        "search_plan": None,
        "max_depth": 2,
        "retrieval_results": [],
        "tool_calls": [],
        "batch_analyses": [],
        "analysis_result": "",
        "validation_results": ["request schema valid", f"role accepted: {request.role}"],
        "final_answer": "",
        "memory_updates": [memory_message],
        "activity_log": [
            {
                "node": "memory",
                "status": "completed",
                "message": memory_message,
            }
        ],
        "errors": [],
        "citations": [],
    }


def activity_from_state(state: AssistantState) -> AgentActivity:
    """Convert LangGraph state into the API sidebar activity model."""

    graph_activities = [
        activity for activity in state["activity_log"] if activity["node"] != "memory_node"
    ]
    last_activity = graph_activities[-1] if graph_activities else None
    active_node = last_activity["node"] if last_activity else "unknown"
    retrieval_count = len(state["retrieval_results"])
    search_plan = state["search_plan"]
    planning_summary = []
    if search_plan is not None:
        planning_summary = [
            f"objective: {search_plan['objective']}",
            f"sub_queries: {len(search_plan['sub_queries'])}",
            f"batch_strategy: {search_plan['batch_strategy']}",
            f"aggregation_strategy: {search_plan['aggregation_strategy']}",
        ]

    return AgentActivity(
        current_state="blocked" if state["errors"] else "completed_langgraph_run",
        active_node=active_node,
        tool_calls=state["tool_calls"],
        retrieval_status=f"completed: retrieved {retrieval_count} chunk(s)",
        validation_results=state["validation_results"] + planning_summary,
        memory_updates=state["memory_updates"],
        activity_log=state["activity_log"],
        errors=state["errors"],
    )


async def run_assistant(request: ChatRequest) -> ChatResponse:
    """Run the LangGraph assistant workflow and return the API response."""

    final_state: AssistantState = await assistant_graph.ainvoke(initial_state_from_request(request))
    final_state = persist_memory_after_run(final_state)

    return ChatResponse(
        answer=final_state["final_answer"],
        session_id=final_state["session_id"],
        agent_activity=activity_from_state(final_state),
        citations=final_state["citations"],
    )
