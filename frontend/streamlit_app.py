"""Streamlit frontend for the BankOps AI Assistant demo."""

import json
import os
from uuid import uuid4

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")
BACKEND_STREAM_URL = os.getenv(
    "BACKEND_STREAM_URL",
    f"{BACKEND_URL.rstrip('/')}/stream" if BACKEND_URL.rstrip("/").endswith("/chat") else BACKEND_URL,
)


st.set_page_config(page_title="BankOps AI Assistant", layout="centered")
st.title("BankOps AI Assistant")
st.caption("Assignment scaffold: FastAPI + Streamlit + LangGraph + Pinecone + LangSmith")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid4()}"
if "agent_activity" not in st.session_state:
    st.session_state.agent_activity = None
if "citations" not in st.session_state:
    st.session_state.citations = []


def show_sidebar() -> str:
    """Render backend agent activity in a compact sidebar panel."""

    st.sidebar.header("Run Context")
    selected_role = st.sidebar.selectbox(
        "User role",
        options=["viewer", "analyst", "administrator"],
        index=1,
    )
    st.sidebar.caption(f"Session: `{st.session_state.session_id}`")
    st.sidebar.caption(f"Backend: `{BACKEND_URL}`")
    st.sidebar.caption(f"Stream: `{BACKEND_STREAM_URL}`")

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.agent_activity = None
        st.session_state.citations = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Agent Activity")

    activity = st.session_state.agent_activity
    if not activity:
        st.sidebar.info("Send a message to see agent activity.")
        return selected_role

    st.sidebar.text_input(
        "Current state",
        value=activity.get("current_state", "unknown"),
        disabled=True,
    )
    st.sidebar.text_input(
        "Active node",
        value=activity.get("active_node", "unknown"),
        disabled=True,
    )
    st.sidebar.write("Tool calls")
    st.sidebar.json(activity.get("tool_calls", []), expanded=False)
    st.sidebar.text_area(
        "Retrieval status",
        value=activity.get("retrieval_status", "not started"),
        disabled=True,
        height=80,
    )
    st.sidebar.write("Validation results")
    st.sidebar.json(activity.get("validation_results", []), expanded=False)
    st.sidebar.write("Memory updates")
    st.sidebar.json(activity.get("memory_updates", []), expanded=False)
    st.sidebar.write("Activity log")
    st.sidebar.json(activity.get("activity_log", []), expanded=False)

    if activity.get("errors"):
        st.sidebar.error("Graph reported errors")
        st.sidebar.json(activity.get("errors", []), expanded=False)

    if st.session_state.citations:
        st.sidebar.divider()
        st.sidebar.write("Citations")
        st.sidebar.json(st.session_state.citations, expanded=False)

    return selected_role


selected_role = show_sidebar()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about enterprise docs, accounts, or next actions")


def build_chat_payload(message: str, role: str) -> dict[str, str]:
    """Build the backend chat request payload."""

    return {
        "user_id": f"{role}-demo-user",
        "role": role,
        "message": message,
        "session_id": st.session_state.session_id,
    }


def call_non_streaming_chat(payload: dict[str, str]) -> str:
    """Call the stable non-streaming endpoint as a fallback."""

    response = requests.post(BACKEND_URL, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    st.session_state.agent_activity = data.get("agent_activity")
    st.session_state.citations = data.get("citations", [])
    return data["answer"]


def stream_chat_response(payload: dict[str, str]) -> str:
    """Stream answer tokens from the backend and update Streamlit placeholders."""

    answer_parts: list[str] = []
    answer_placeholder = st.empty()
    live_activity = st.sidebar.empty()

    with requests.post(
        BACKEND_STREAM_URL,
        json=payload,
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            event = json.loads(raw_line)
            event_type = event.get("type")
            data = event.get("data", {})

            if event_type == "activity_update":
                st.session_state.agent_activity = data
                live_activity.json(data, expanded=False)
            elif event_type == "token":
                answer_parts.append(str(data))
                answer_placeholder.markdown("".join(answer_parts) + "▌")
            elif event_type == "final_metadata":
                st.session_state.agent_activity = data.get("agent_activity")
                st.session_state.citations = data.get("citations", [])
            elif event_type == "error":
                raise RuntimeError(str(data.get("message", "Streaming failed.")))

    answer = "".join(answer_parts).strip()
    answer_placeholder.markdown(answer)
    return answer

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            payload = build_chat_payload(prompt, selected_role)
            try:
                answer = stream_chat_response(payload)
            except (requests.RequestException, RuntimeError, json.JSONDecodeError):
                answer = call_non_streaming_chat(payload)
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            answer = f"Backend returned an error: `{detail}`"
            st.session_state.agent_activity = None
            st.session_state.citations = []
        except requests.RequestException:
            answer = (
                "I could not reach the backend. Start it with "
                "`uvicorn app.main:app --reload --app-dir backend`."
            )
            st.session_state.agent_activity = None
            st.session_state.citations = []

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
