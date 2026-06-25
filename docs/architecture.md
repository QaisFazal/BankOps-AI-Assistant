# Architecture

This project is a starter scaffold for an enterprise AI assistant.

## Components

- **Frontend:** Streamlit chat UI for demos and assignment walkthroughs.
- **Backend:** FastAPI app exposing health and chat endpoints.
- **Orchestration:** LangGraph will coordinate the assistant workflow.
- **Retrieval:** Pinecone will store and search embedded enterprise documents.
- **Tracing:** LangSmith will capture runs, errors, and evaluation traces.
- **Security:** Guardrail placeholders mark where auth, authorization, and data policy checks belong.
- **Memory:** Session-based in-memory conversation history stores recent questions and answers.

## Planned Request Flow

1. A user asks a question in Streamlit.
2. Streamlit sends the message to the FastAPI `/chat` endpoint.
3. The backend checks security policy and loads conversation memory.
4. LangGraph loads session memory, routes through safety checks, planning, retrieval, analysis, response generation, and citation validation.
5. Local hybrid retrieval returns JSONL chunks by default; Pinecone can be enabled later by configuration.
6. LangSmith records traces for debugging and evaluation.
7. The backend stores the latest question and answer in session memory.
8. The backend returns an answer, citations, and agent activity for the Streamlit sidebar.

## Memory Design Tradeoffs

The current memory store is process-local and keyed by `session_id`. It is simple,
fast, and easy to test, which makes it useful for the assignment prototype. It
also means memory is lost when the backend restarts and is not shared across
multiple backend processes.

To keep memory from growing indefinitely, older turns are summarized after a
small threshold and only the most recent turns are kept verbatim. The summary is
deterministic and lightweight rather than LLM-generated, so it is predictable but
less nuanced than a production summarizer.

For production, this should move to durable storage such as Postgres, Redis, or a
managed memory service with encryption, retention controls, tenant isolation,
and user deletion support.

## Current State

The assistant still uses template-based answer generation. Retrieval, planning,
memory, citation validation, and activity logging are implemented, but LLM-based
generation is intentionally not wired yet.
