"""LangGraph workflow for the assistant."""

import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.memory.store import (
    get_active_topic,
    get_conversation_history,
    save_conversation_turn,
)
from app.models.chat import AgentActivity, ChatRequest, ChatResponse, Citation
from app.models.documents import RetrievedDocument
from app.observability.tracing import build_run_metadata, set_trace_outputs, trace_run
from app.retrieval.base import MetadataFilter
from app.security.guardrails import (
    GuardrailViolation,
    detect_prompt_injection,
    validate_user_question,
)
from app.services.contextualization_service import (
    ContextualizedQuestion,
    contextualize_question,
)
from app.services.llm_service import generate_answer as generate_llm_answer
from app.tools.dummy_mcp import dummy_mcp_tool
from app.tools.errors import ToolExecutionError, ToolTimeoutError
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
    standalone_question: str
    conversation_history: list[str]
    active_topic: str | None
    contextualization_confidence: float
    contextualization_source: str
    needs_clarification: bool
    clarification_question: str | None
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

RESTRICTED_OPERATION_MARKERS = {
    "card network failover": "Card network failover procedures are restricted operational content.",
}

FALLBACK_FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|they|their|them|that|this|those|these)\b"
    r"|^(what|which|why|how|when|who)\b",
    re.IGNORECASE,
)

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DOCS_ROOT = ROOT / "sample_docs"
logger = logging.getLogger(__name__)


def fallback_contextualization(
    question: str,
    conversation_history: list[str],
    active_topic: str | None,
) -> ContextualizedQuestion:
    """Provide a small deterministic fallback when Gemini rewriting is unavailable."""

    if not conversation_history:
        return ContextualizedQuestion(
            is_follow_up=False,
            standalone_question=question,
            active_topic=question[:240],
            confidence=1.0,
        )

    inferred_topic = active_topic
    if not inferred_topic:
        recent_user_questions = [
            item.removeprefix("User: ").strip()
            for item in conversation_history
            if item.startswith("User: ")
        ]
        inferred_topic = recent_user_questions[-1] if recent_user_questions else None

    is_follow_up = bool(inferred_topic and FALLBACK_FOLLOW_UP_PATTERN.search(question))
    if is_follow_up:
        return ContextualizedQuestion(
            is_follow_up=True,
            standalone_question=f"{question} Subject: {inferred_topic}",
            active_topic=inferred_topic,
            confidence=0.8,
        )

    return ContextualizedQuestion(
        is_follow_up=False,
        standalone_question=question,
        active_topic=question[:240],
        confidence=0.8,
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


def requested_mcp_resource(question: str) -> str | None:
    """Infer the dummy MCP resource requested by the user, if any."""

    lowered = question.lower()
    if "employee directory" in lowered:
        return "employee_directory"
    if "service catalog" in lowered:
        return "service_catalog"
    if "incident records" in lowered:
        return "incident_records"
    if "mcp" in lowered or "enterprise data" in lowered:
        return "service_catalog"

    return None


def requested_restricted_operation(question: str) -> str | None:
    """Return a restricted operation marker requested by the user, if any."""

    lowered = question.lower()
    for marker, message in RESTRICTED_OPERATION_MARKERS.items():
        if marker in lowered:
            return message

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
            user_id=state["user_id"],
            session_id=state["session_id"],
        )
    except ToolTimeoutError as exc:
        state["errors"].append(f"Tool timed out: {exc}")
        state["validation_results"].append("tool execution failed")
        append_activity(
            state,
            "knowledge_search_tool",
            "failed",
            f"Depth {depth}: search tool timed out.",
        )
        return []
    except (ToolPermissionError, GuardrailViolation, ToolExecutionError) as exc:
        state["errors"].append(f"Tool guardrail blocked call: {exc}")
        state["validation_results"].append("tool guardrail failed")
        append_activity(
            state,
            "guardrails",
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


def memory_key_for_user(session_id: str, user_id: str, role: str) -> str:
    """Scope memory by session, user, and role to avoid access-level leaks."""

    return f"{session_id}:{user_id}:{role.lower()}"


def run_optional_mcp_tool(state: AssistantState) -> list[dict[str, Any]]:
    """Try to load optional enterprise context without blocking the graph."""

    resource = requested_mcp_resource(state["question"])
    if resource is None:
        return []

    state["tool_calls"].append("dummy_mcp_tool")
    try:
        records = dummy_mcp_tool(
            resource,
            role=state["role"],
            user_id=state["user_id"],
            session_id=state["session_id"],
        )
    except ToolPermissionError as exc:
        state["errors"].append(f"Unauthorized: role {state['role']} cannot execute dummy_mcp_tool.")
        state["validation_results"].append("tool authorization failed")
        append_activity(state, "dummy_mcp_tool", "failed", "MCP authorization failed.")
        raise exc
    except Exception as exc:
        state["errors"].append("MCP tool failed; continued without MCP context.")
        state["validation_results"].append("mcp tool failed")
        logger.exception(
            "MCP tool failed; continuing without MCP context",
            extra={
                "component": "tool",
                "operation": "dummy_mcp_tool",
                "error_type": type(exc).__name__,
                "fallback": "continue_without_mcp",
                "user_id": state["user_id"],
                "role": state["role"],
                "session_id": state["session_id"],
                "tool_name": "dummy_mcp_tool",
            },
        )
        append_activity(
            state,
            "dummy_mcp_tool",
            "failed",
            "MCP data was unavailable, so the graph continued with document retrieval only.",
        )
        return []

    state["analysis_result"] = (
        f"MCP context loaded from {resource}: {len(records)} record(s). "
        + state["analysis_result"]
    ).strip()
    state["validation_results"].append("mcp tool completed")
    append_activity(
        state,
        "dummy_mcp_tool",
        "completed",
        f"Loaded {len(records)} record(s) from {resource}.",
    )
    return records


async def supervisor_node(state: AssistantState) -> AssistantState:
    """Classify the standalone question after safety and contextualization."""

    state["intent"] = classify_intent(state["standalone_question"])
    history_count = len(state["conversation_history"])
    return append_activity(
        state,
        "supervisor_node",
        "completed",
        (
            f"Classified intent as {state['intent']} and routed to planning_node. "
            f"Loaded {history_count} memory item(s)."
        ),
    )


async def security_node(state: AssistantState) -> AssistantState:
    """Detect prompt injection and tool abuse attempts."""

    try:
        validate_user_question(state["question"])
    except GuardrailViolation as exc:
        state["errors"].append(f"Input validation failed: {exc}")
        state["validation_results"].append("input length validation failed")
        append_activity(state, "guardrails", "blocked", str(exc))
        return append_activity(
            state,
            "security_node",
            "blocked",
            "Blocked request because user input failed validation.",
        )

    append_activity(
        state,
        "guardrails",
        "passed",
        "User input length validation passed.",
    )

    suspicious_markers = detect_prompt_injection(state["question"])

    if suspicious_markers:
        state["errors"].append(
            "Potential prompt injection or tool abuse detected: "
            + ", ".join(suspicious_markers)
        )
        state["validation_results"].append("security check failed")
        append_activity(
            state,
            "guardrails",
            "blocked",
            "Prompt injection marker(s): " + ", ".join(suspicious_markers),
        )
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
        append_activity(
            state,
            "guardrails",
            "blocked",
            f"Unauthorized request for {restricted_tool}.",
        )
        return append_activity(
            state,
            "security_node",
            "blocked",
            f"Blocked unauthorized request for {restricted_tool}.",
        )

    restricted_operation = requested_restricted_operation(state["question"])
    if restricted_operation and state["role"] != "administrator":
        state["errors"].append(
            f"Unauthorized: role {state['role']} cannot access restricted operation details."
        )
        state["validation_results"].append("restricted operation authorization failed")
        append_activity(
            state,
            "guardrails",
            "blocked",
            restricted_operation,
        )
        return append_activity(
            state,
            "security_node",
            "blocked",
            "Blocked restricted operational procedure request before retrieval.",
        )

    state["validation_results"].append("security check passed")
    append_activity(
        state,
        "guardrails",
        "passed",
        "Prompt injection and tool authorization guardrails passed.",
    )
    return append_activity(
        state,
        "security_node",
        "completed",
        "No prompt injection or tool abuse markers detected.",
    )


def route_after_security(
    state: AssistantState,
) -> Literal["contextualization_node", "response_node"]:
    """Skip all model and retrieval work when security blocks the original request."""

    if state["errors"]:
        return "response_node"

    return "contextualization_node"


async def contextualization_node(state: AssistantState) -> AssistantState:
    """Rewrite conversational follow-ups into safe standalone retrieval questions."""

    result = await contextualize_question(
        question=state["question"],
        conversation_history=state["conversation_history"],
        active_topic=state["active_topic"],
        user_id=state["user_id"],
        role=state["role"],
        session_id=state["session_id"],
    )
    source = "gemini" if result is not None else "deterministic_fallback"
    if result is None:
        result = fallback_contextualization(
            state["question"],
            state["conversation_history"],
            state["active_topic"],
        )

    try:
        validate_user_question(result.standalone_question)
        if detect_prompt_injection(result.standalone_question):
            raise GuardrailViolation("Contextualized question failed security validation.")
    except GuardrailViolation:
        result = fallback_contextualization(
            state["question"],
            state["conversation_history"],
            state["active_topic"],
        )
        source = "security_validated_fallback"

    state["standalone_question"] = result.standalone_question
    state["active_topic"] = result.active_topic
    state["contextualization_confidence"] = result.confidence
    state["contextualization_source"] = source

    restricted_tool = requested_restricted_tool(result.standalone_question)
    if restricted_tool and not can_execute_tool(restricted_tool, state["role"]):
        state["errors"].append(
            f"Unauthorized: role {state['role']} cannot execute {restricted_tool}."
        )
        state["validation_results"].append("contextualized tool authorization failed")
        return append_activity(
            state,
            "contextualization_node",
            "blocked",
            f"Standalone question requested unauthorized tool {restricted_tool}.",
        )

    restricted_operation = requested_restricted_operation(result.standalone_question)
    if restricted_operation and state["role"] != "administrator":
        state["errors"].append(
            f"Unauthorized: role {state['role']} cannot access restricted operation details."
        )
        state["validation_results"].append(
            "contextualized restricted operation authorization failed"
        )
        return append_activity(
            state,
            "contextualization_node",
            "blocked",
            restricted_operation,
        )

    threshold = get_settings().contextualization_confidence_threshold
    if result.confidence < threshold:
        state["needs_clarification"] = True
        state["clarification_question"] = result.clarification_question or (
            "Could you clarify which incident, document, or service you mean?"
        )
        state["validation_results"].append("contextualization needs clarification")
        return append_activity(
            state,
            "contextualization_node",
            "needs_clarification",
            f"Rewrite confidence {result.confidence:.2f} was below {threshold:.2f}.",
        )

    state["validation_results"].append(
        f"contextualization completed via {source}"
    )
    if result.is_follow_up:
        state["validation_results"].append("follow-up query contextualized from memory")
    return append_activity(
        state,
        "contextualization_node",
        "completed",
        (
            f"Prepared standalone retrieval question with confidence "
            f"{result.confidence:.2f} via {source}."
        ),
    )


def route_after_contextualization(
    state: AssistantState,
) -> Literal["supervisor_node", "response_node"]:
    """Request clarification without running retrieval when the subject is ambiguous."""

    if state["errors"] or state["needs_clarification"]:
        return "response_node"
    return "supervisor_node"


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

    state["search_plan"] = generate_search_plan(
        state["standalone_question"],
        state["intent"],
    )
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

    try:
        run_optional_mcp_tool(state)
    except ToolPermissionError as exc:
        state["errors"].append(f"Tool guardrail blocked call: {exc}")
        return append_activity(
            state,
            "retrieval_node",
            "blocked",
            str(exc),
        )

    if state["search_plan"] is None:
        state["tool_calls"].append("knowledge_search_tool")
        try:
            state["retrieval_results"] = await knowledge_search_tool(
                state["standalone_question"],
                role=state["role"],
                top_k=3,
                user_id=state["user_id"],
                session_id=state["session_id"],
            )
        except ToolTimeoutError as exc:
            state["errors"].append(f"Tool timed out: {exc}")
            state["validation_results"].append("tool execution failed")
            state["retrieval_results"] = []
            append_activity(
                state,
                "knowledge_search_tool",
                "failed",
                "Search tool timed out before retrieval completed.",
            )
            return append_activity(
                state,
                "retrieval_node",
                "failed",
                "Retrieval could not complete because the search tool timed out.",
            )
        except (ToolPermissionError, GuardrailViolation, ToolExecutionError) as exc:
            state["errors"].append(f"Tool guardrail blocked call: {exc}")
            state["validation_results"].append("tool guardrail failed")
            state["retrieval_results"] = []
            append_activity(state, "guardrails", "blocked", str(exc))
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


def normalize_markdown_text(text: str) -> str:
    """Make retrieved markdown sections readable inside a chat answer."""

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped.removeprefix("- ").strip())

    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def truncate_sentence(text: str, max_chars: int = 260) -> str:
    """Trim text without leaving a noisy half-sentence when possible."""

    clean_text = normalize_markdown_text(text)
    if len(clean_text) <= max_chars:
        return clean_text

    truncated = clean_text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def extract_markdown_section(markdown: str, section_names: tuple[str, ...]) -> str:
    """Extract a named markdown section from a retrieved chunk."""

    active = False
    section_lines: list[str] = []
    wanted = {name.lower() for name in section_names}

    for line in markdown.splitlines():
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if active and heading not in wanted:
                break
            active = heading in wanted
            continue

        if active:
            section_lines.append(line)

    return "\n".join(section_lines).strip()


def fallback_summary(markdown: str) -> str:
    """Return the first useful paragraph when a named section is not present."""

    paragraphs = [
        paragraph.strip()
        for paragraph in markdown.split("\n\n")
        if paragraph.strip() and not paragraph.strip().startswith("#")
    ]
    return paragraphs[0] if paragraphs else markdown


def read_source_body(result: RetrievedDocument) -> str:
    """Load the original markdown body for a retrieved source when available."""

    if not result.source_file:
        return result.snippet

    source_path = (SAMPLE_DOCS_ROOT / result.source_file).resolve()
    try:
        source_path.relative_to(SAMPLE_DOCS_ROOT.resolve())
    except ValueError:
        return result.snippet

    if not source_path.exists():
        return result.snippet

    text = source_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()

    return text.strip()


def source_reference(result: RetrievedDocument) -> str:
    """Create a compact citation reference for one retrieved chunk."""

    parts = [result.source_file or result.title]
    if result.chunk_id:
        parts.append(f"chunk {result.chunk_id}")
    return ", ".join(parts)


def unique_answer_sources(results: list[RetrievedDocument], limit: int = 4) -> list[RetrievedDocument]:
    """Keep one answer card per source document while preserving retrieval rank."""

    seen: set[str] = set()
    unique_results: list[RetrievedDocument] = []
    for result in results:
        key = result.document_id or result.source_file or result.title
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(result)
        if len(unique_results) == limit:
            break

    return unique_results


def format_retrieved_source(result: RetrievedDocument, index: int) -> str:
    """Format one retrieved document as a useful answer section."""

    source_body = read_source_body(result)
    summary = extract_markdown_section(source_body, ("Summary",)) or fallback_summary(
        source_body
    )
    impact = extract_markdown_section(source_body, ("Impact", "Customer Impact"))
    root_cause = extract_markdown_section(source_body, ("Root Cause",))
    resolution = extract_markdown_section(source_body, ("Resolution",))

    metadata_bits = [
        bit
        for bit in (result.department, result.document_type, result.created_date)
        if bit
    ]
    metadata = f" ({', '.join(metadata_bits)})" if metadata_bits else ""
    lines = [
        f"{index}. **{result.title}**{metadata}",
        f"   - Summary: {truncate_sentence(summary)}",
    ]

    if impact:
        lines.append(f"   - Impact: {truncate_sentence(impact, max_chars=220)}")
    if root_cause:
        lines.append(f"   - Root cause: {truncate_sentence(root_cause, max_chars=220)}")
    if resolution:
        lines.append(f"   - Resolution: {truncate_sentence(resolution, max_chars=220)}")

    lines.append(f"   - Source: `{source_reference(result)}`")
    return "\n".join(lines)


def build_synthesis_note(
    retrieval_results: list[RetrievedDocument],
    analysis_result: str,
) -> str:
    """Add a short cross-source takeaway without exposing internal planning logs."""

    if not retrieval_results:
        return ""

    departments = sorted({result.department for result in retrieval_results if result.department})
    doc_types = sorted({result.document_type for result in retrieval_results if result.document_type})
    scope_parts = []
    if departments:
        scope_parts.append(f"departments: {', '.join(departments)}")
    if doc_types:
        scope_parts.append(f"document types: {', '.join(doc_types)}")

    scope_text = f" ({'; '.join(scope_parts)})" if scope_parts else ""
    if analysis_result.startswith("Aggregated planned search batches"):
        source_count = len(unique_answer_sources(retrieval_results, limit=100))
        return f"\n\nTakeaway: I found {source_count} relevant source document(s){scope_text}."

    return f"\n\nTakeaway: {analysis_result}{scope_text}."


def has_blocking_security_error(errors: list[str]) -> bool:
    """Return whether errors represent unsafe user input or authorization failure."""

    blocking_markers = (
        "Potential prompt injection",
        "Input validation failed",
        "Unauthorized:",
        "Citation validation failed",
        "Tool guardrail blocked call",
    )
    return any(error.startswith(blocking_markers) for error in errors)


def should_use_llm_answer(state: AssistantState) -> bool:
    """Return whether the response node should try Gemini generation."""

    if not state["retrieval_results"] or not state["citations"]:
        return False
    if has_blocking_security_error(state["errors"]):
        return False
    if any("timed out" in error.lower() for error in state["errors"]):
        return False

    return True


def build_answer(
    question: str,
    citations: list[Citation],
    errors: list[str],
    analysis_result: str,
    search_plan: SearchPlan | None,
    conversation_history: list[str],
    retrieval_results: list[RetrievedDocument],
) -> str:
    """Build a simple retrieval-grounded final answer."""

    if errors:
        if any(error.startswith("Unauthorized:") or "Unauthorized tool call" in error for error in errors):
            if any("restricted operation details" in error for error in errors):
                return (
                    "I cannot provide those steps because they are restricted operational "
                    "procedures. Please contact cards operations or use an administrator "
                    "role if you are authorized."
                )

            return (
                "You are not authorized to use the requested tool. "
                "Viewers can chat and search approved knowledge only. "
                "Please ask a document-search question or switch to an authorized role."
            )

        if any("MCP tool failed" in error for error in errors):
            limitation = (
                "Note: the enterprise MCP data source was unavailable, so I continued "
                "using the approved document search results only."
            )
            if not citations:
                return (
                    limitation
                    + " I could not find enough local document context to answer confidently."
                )
        elif any("timed out" in error.lower() for error in errors):
            return (
                "I could not complete the request because a backend tool timed out. "
                "Please try again in a moment."
            )
        elif any("response generation failed" in error for error in errors):
            return (
                "I found relevant context, but the answer generator failed. "
                "Please try again in a moment."
            )
        elif has_blocking_security_error(errors):
            return (
                "I cannot answer that request because it appears to contain unsafe instructions. "
                "Please ask a normal question about the approved enterprise documents."
            )
        else:
            return (
                "I hit a temporary backend issue while processing that request. "
                "Please try again in a moment."
            )
    else:
        limitation = ""

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

    answer_sources = unique_answer_sources(retrieval_results)
    source_summaries = [
        format_retrieved_source(result, index)
        for index, result in enumerate(answer_sources, start=1)
    ]
    scope_note = ""
    if search_plan is not None:
        scope_note = (
            "\n\nI used a multi-step search plan to compare related incident documents, "
            "then deduplicated the strongest sources."
        )

    return (
        f"Here is a grounded summary for: {question}\n\n"
        + "\n".join(source_summaries)
        + build_synthesis_note(retrieval_results, analysis_result)
        + scope_note
        + (f"\n\n{limitation}" if limitation else "")
        + memory_note
        + "\n\nSources were validated against retrieved chunks before this answer was returned."
    )


async def response_node(state: AssistantState) -> AssistantState:
    """Generate the final answer with citations."""

    if state["needs_clarification"]:
        state["citations"] = []
        state["final_answer"] = state["clarification_question"] or (
            "Could you clarify which subject you mean?"
        )
        return append_activity(
            state,
            "response_node",
            "needs_clarification",
            "Returned a clarification question without running retrieval.",
        )

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
    try:
        deterministic_answer = build_answer(
            state["question"],
            state["citations"],
            state["errors"],
            state["analysis_result"],
            state["search_plan"],
            state["conversation_history"],
            state["retrieval_results"],
        )
        llm_answer = None
        if should_use_llm_answer(state):
            llm_answer = await generate_llm_answer(
                question=state["question"],
                role=state["role"],
                user_id=state["user_id"],
                session_id=state["session_id"],
                retrieval_results=state["retrieval_results"],
                citations=state["citations"],
                conversation_history=state["conversation_history"],
                analysis_result=state["analysis_result"],
                errors=state["errors"],
            )

        if llm_answer:
            state["final_answer"] = llm_answer
            state["validation_results"].append("gemini generation completed")
            append_activity(
                state,
                "response_node",
                "completed",
                f"Generated Gemini answer with {len(state['citations'])} citation(s).",
            )
            return state

        state["final_answer"] = deterministic_answer
        if should_use_llm_answer(state):
            state["validation_results"].append("gemini fallback used")
            append_activity(
                state,
                "response_node",
                "completed",
                "Gemini was unavailable, so deterministic fallback answer was used.",
            )
            return state
    except Exception as exc:
        state["errors"].append("response generation failed")
        state["validation_results"].append("response generation failed")
        state["final_answer"] = (
            "I found relevant context, but the answer generator failed. "
            "Please try again in a moment."
        )
        logger.exception(
            "Response generation failed",
            extra={
                "component": "agent",
                "operation": "response_node",
                "error_type": type(exc).__name__,
                "fallback": "safe_llm_fallback_message",
                "user_id": state["user_id"],
                "role": state["role"],
                "session_id": state["session_id"],
            },
        )
        append_activity(
            state,
            "response_node",
            "failed",
            "Answer generation failed, so a fallback message was returned.",
        )
        return state

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
        append_activity(
            state,
            "guardrails",
            "blocked",
            "Citation validation failed for non-retrieved chunk id(s).",
        )
        return append_activity(
            state,
            "citation_validation_node",
            "failed",
            "One or more citations did not map to retrieved chunks.",
        )

    state["validation_results"].append("citation validation passed")
    append_activity(
        state,
        "guardrails",
        "passed",
        "Citation validation guardrail passed.",
    )
    return append_activity(
        state,
        "citation_validation_node",
        "completed",
        "All citations map to retrieved chunks.",
    )


def persist_memory_after_run(state: AssistantState) -> AssistantState:
    """Save the completed user/assistant turn to session memory."""

    memory_key = memory_key_for_user(state["session_id"], state["user_id"], state["role"])
    updates = save_conversation_turn(
        session_id=memory_key,
        question=state["question"],
        answer=state["final_answer"],
        standalone_question=state["standalone_question"],
        active_topic=state["active_topic"],
    )
    state["memory_updates"].extend(updates)
    state["conversation_history"] = get_conversation_history(memory_key)

    for update in updates:
        append_activity(state, "memory_node", "updated", update)

    return state


def build_assistant_graph():
    """Build and compile the assistant LangGraph workflow."""

    graph = StateGraph(AssistantState)
    graph.add_node("security_node", security_node)
    graph.add_node("contextualization_node", contextualization_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("response_node", response_node)
    graph.add_node("citation_validation_node", citation_validation_node)

    graph.add_edge(START, "security_node")
    graph.add_conditional_edges(
        "security_node",
        route_after_security,
        {
            "contextualization_node": "contextualization_node",
            "response_node": "response_node",
        },
    )
    graph.add_conditional_edges(
        "contextualization_node",
        route_after_contextualization,
        {
            "supervisor_node": "supervisor_node",
            "response_node": "response_node",
        },
    )
    graph.add_edge("supervisor_node", "planning_node")
    graph.add_edge("planning_node", "retrieval_node")
    graph.add_edge("retrieval_node", "analysis_node")
    graph.add_edge("analysis_node", "response_node")
    graph.add_edge("response_node", "citation_validation_node")
    graph.add_edge("citation_validation_node", END)

    return graph.compile()


assistant_graph = build_assistant_graph()


def initial_state_from_request(request: ChatRequest) -> AssistantState:
    """Create LangGraph state from an API chat request."""

    memory_key = memory_key_for_user(request.session_id, request.user_id, request.role)
    conversation_history = get_conversation_history(memory_key)
    active_topic = get_active_topic(memory_key)
    memory_message = (
        f"Loaded {len(conversation_history)} memory item(s) for this user and role."
        if conversation_history
        else f"No stored memory yet for this user and role in session {request.session_id}."
    )
    return {
        "user_id": request.user_id,
        "role": request.role,
        "session_id": request.session_id,
        "question": request.message,
        "standalone_question": request.message,
        "conversation_history": conversation_history,
        "active_topic": active_topic,
        "contextualization_confidence": 1.0,
        "contextualization_source": "not_run",
        "needs_clarification": False,
        "clarification_question": None,
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

    contextualization_summary = [
        f"contextualization_source: {state['contextualization_source']}",
        f"contextualization_confidence: {state['contextualization_confidence']:.2f}",
        f"standalone_question: {state['standalone_question']}",
    ]

    if state["needs_clarification"]:
        current_state = "needs_clarification"
    elif state["errors"] and has_blocking_security_error(state["errors"]):
        current_state = "blocked"
    elif state["errors"]:
        current_state = "completed_with_warnings"
    else:
        current_state = "completed_langgraph_run"

    return AgentActivity(
        current_state=current_state,
        active_node=active_node,
        tool_calls=state["tool_calls"],
        retrieval_status=(
            "not run: clarification required"
            if state["needs_clarification"]
            else f"completed: retrieved {retrieval_count} chunk(s)"
        ),
        validation_results=(
            state["validation_results"]
            + contextualization_summary
            + planning_summary
        ),
        memory_updates=state["memory_updates"],
        activity_log=state["activity_log"],
        errors=state["errors"],
    )


async def run_assistant(request: ChatRequest) -> ChatResponse:
    """Run the LangGraph assistant workflow and return the API response."""

    metadata = build_run_metadata(
        user_id=request.user_id,
        role=request.role,
        session_id=request.session_id,
        graph_name="assistant_graph",
    )
    with trace_run(
        "langgraph_assistant_run",
        run_type="chain",
        inputs={"question": request.message},
        metadata=metadata,
        tags=["langgraph", "chat"],
    ) as run:
        try:
            final_state: AssistantState = await assistant_graph.ainvoke(
                initial_state_from_request(request)
            )
            final_state = persist_memory_after_run(final_state)

            response = ChatResponse(
                answer=final_state["final_answer"],
                session_id=final_state["session_id"],
                agent_activity=activity_from_state(final_state),
                citations=final_state["citations"],
            )
        except Exception as exc:
            logger.exception(
                "LangGraph assistant run failed",
                extra={
                    "component": "agent",
                    "operation": "langgraph_assistant_run",
                    "error_type": type(exc).__name__,
                    "fallback": "safe_chat_response",
                    "user_id": request.user_id,
                    "role": request.role,
                    "session_id": request.session_id,
                },
            )
            response = ChatResponse(
                answer=(
                    "I hit a temporary backend issue while processing that request. "
                    "Please try again in a moment."
                ),
                session_id=request.session_id,
                agent_activity=AgentActivity(
                    current_state="completed_with_warnings",
                    active_node="langgraph_assistant_run",
                    tool_calls=[],
                    retrieval_status="not completed",
                    validation_results=["graph execution failed"],
                    memory_updates=[],
                    activity_log=[
                        {
                            "node": "langgraph_assistant_run",
                            "status": "failed",
                            "message": "Assistant graph failed, so a fallback response was returned.",
                        }
                    ],
                    errors=["graph execution failed"],
                ),
                citations=[],
            )

        set_trace_outputs(
            run,
            {
                "current_state": response.agent_activity.current_state,
                "active_node": response.agent_activity.active_node,
                "tool_calls": response.agent_activity.tool_calls,
                "citation_count": len(response.citations),
                "error_count": len(response.agent_activity.errors),
            },
        )
        return response


async def stream_assistant_events(request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
    """Stream graph activity, final answer tokens, and final metadata."""

    metadata = build_run_metadata(
        user_id=request.user_id,
        role=request.role,
        session_id=request.session_id,
        graph_name="assistant_graph",
        streaming=True,
    )
    with trace_run(
        "langgraph_assistant_stream",
        run_type="chain",
        inputs={"question": request.message},
        metadata=metadata,
        tags=["langgraph", "chat", "streaming"],
    ) as run:
        try:
            final_state: AssistantState | None = None
            initial_state = initial_state_from_request(request)
            yield {
                "type": "activity_update",
                "data": activity_from_state(initial_state).model_dump(),
            }

            async for state_update in assistant_graph.astream(
                initial_state,
                stream_mode="values",
            ):
                final_state = state_update
                yield {
                    "type": "activity_update",
                    "data": activity_from_state(final_state).model_dump(),
                }

            if final_state is None:
                raise RuntimeError("Assistant graph did not produce a final state.")

            final_state = persist_memory_after_run(final_state)
            response = ChatResponse(
                answer=final_state["final_answer"],
                session_id=final_state["session_id"],
                agent_activity=activity_from_state(final_state),
                citations=final_state["citations"],
            )
            yield {
                "type": "activity_update",
                "data": response.agent_activity.model_dump(),
            }

            for token in re.findall(r"\S+\s*", response.answer):
                yield {"type": "token", "data": token}

            yield {
                "type": "final_metadata",
                "data": {
                    "session_id": response.session_id,
                    "agent_activity": response.agent_activity.model_dump(),
                    "citations": [citation.model_dump() for citation in response.citations],
                },
            }
            set_trace_outputs(
                run,
                {
                    "current_state": response.agent_activity.current_state,
                    "active_node": response.agent_activity.active_node,
                    "tool_calls": response.agent_activity.tool_calls,
                    "citation_count": len(response.citations),
                    "error_count": len(response.agent_activity.errors),
                    "streamed": True,
                },
            )
        except Exception as exc:
            logger.exception(
                "Streaming assistant run failed",
                extra={
                    "component": "agent",
                    "operation": "langgraph_assistant_stream",
                    "error_type": type(exc).__name__,
                    "fallback": "stream_error_event",
                    "user_id": request.user_id,
                    "role": request.role,
                    "session_id": request.session_id,
                },
            )
            yield {
                "type": "error",
                "data": {
                    "message": (
                        "I hit a temporary backend issue while streaming that request. "
                        "Please try again in a moment."
                    ),
                    "session_id": request.session_id,
                },
            }
            set_trace_outputs(
                run,
                {
                    "current_state": "completed_with_warnings",
                    "active_node": "langgraph_assistant_stream",
                    "error_count": 1,
                    "streamed": True,
                },
            )
