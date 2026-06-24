"""Streamlit frontend for the AI Lead Assistant demo."""

import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/chat")


st.set_page_config(page_title="AI Lead Assistant", layout="centered")
st.title("AI Lead Assistant")
st.caption("Assignment scaffold: FastAPI + Streamlit + LangGraph + Pinecone + LangSmith")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                json={"message": prompt, "session_id": "streamlit-demo"},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
        except requests.RequestException:
            answer = (
                "I could not reach the backend. Start it with "
                "`uvicorn app.main:app --reload --app-dir backend`."
            )

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
