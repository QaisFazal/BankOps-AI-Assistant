"""Streamlit frontend for the AI Lead Assistant demo."""

import os
from uuid import uuid4

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/chat")


st.set_page_config(page_title="AI Lead Assistant", layout="centered")
st.title("AI Lead Assistant")
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

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                BACKEND_URL,
                json={
                    "user_id": f"{selected_role}-demo-user",
                    "role": selected_role,
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
            st.session_state.agent_activity = data.get("agent_activity")
            st.session_state.citations = data.get("citations", [])
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
