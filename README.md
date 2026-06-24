# AI Lead Assistant

Starter Python project for an enterprise AI assistant assignment.

The scaffold uses FastAPI for the backend, Streamlit for the demo frontend,
LangGraph for future orchestration, a Pinecone retrieval abstraction, and
LangSmith tracing placeholders.

## Quick Start

```bash
cd ai-lead-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
python scripts\seed_mock_docs.py
uvicorn app.main:app --reload --app-dir backend
```

In a second terminal:

```bash
streamlit run frontend\streamlit_app.py
```

## What Works Now

- `GET /health` returns a health response.
- `POST /api/chat` returns a scaffolded assistant response.
- Streamlit can send chat messages to the backend.
- Scripts can create and list mock documents.

## What Is Intentionally Not Built Yet

- Real LangGraph workflow nodes.
- Model calls and prompt templates.
- Pinecone index creation, embedding, and upsert logic.
- LangSmith callback wiring.
- Enterprise authentication and authorization.
- Persistent memory.

## Suggested Next Milestones

1. Define the assistant use cases and graph states.
2. Add document chunking and embeddings.
3. Implement Pinecone upsert/query methods behind `retrieval/vector_store.py`.
4. Add LangSmith tracing around graph runs.
5. Add security checks before retrieval and tool execution.
