# Agent Notes

This repository is intentionally scaffold-first.

## Guidance for Future Agents

- Keep examples beginner-friendly and avoid hiding important behavior behind clever abstractions.
- Prefer small modules with clear ownership: API, agents, retrieval, tools, security, memory, observability, models, and services.
- Do not put real secrets in source control. Use `.env` locally and a secret manager in deployed environments.
- Keep Pinecone access behind `backend/app/retrieval/vector_store.py`.
- Keep LangGraph orchestration inside `backend/app/agents`.
- Keep LangSmith setup inside `backend/app/observability`.
- Add tests when replacing placeholders with real behavior.
