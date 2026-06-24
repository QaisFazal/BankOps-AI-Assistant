# Architecture

This project is a starter scaffold for an enterprise AI assistant.

## Components

- **Frontend:** Streamlit chat UI for demos and assignment walkthroughs.
- **Backend:** FastAPI app exposing health and chat endpoints.
- **Orchestration:** LangGraph will coordinate the assistant workflow.
- **Retrieval:** Pinecone will store and search embedded enterprise documents.
- **Tracing:** LangSmith will capture runs, errors, and evaluation traces.
- **Security:** Guardrail placeholders mark where auth, authorization, and data policy checks belong.
- **Memory:** Conversation memory starts as a stub and can later move to a database.

## Planned Request Flow

1. A user asks a question in Streamlit.
2. Streamlit sends the message to the FastAPI `/api/chat` endpoint.
3. The backend checks security policy and loads conversation memory.
4. LangGraph routes the request through retrieval, tools, and response generation.
5. Pinecone returns relevant document chunks.
6. LangSmith records traces for debugging and evaluation.
7. The backend returns an answer, source snippets, and optional debug notes.

## Current State

The scaffold intentionally returns a mock assistant response. This keeps the
assignment easy to review before API keys, embeddings, graph state, and tool
permissions are added.
