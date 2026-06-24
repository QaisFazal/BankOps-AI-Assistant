# Demo Script

## Goal

Show the project structure and prove that the backend/frontend path is ready for
future assistant logic.

## Steps

1. Install dependencies with `pip install -e ".[dev]"`.
2. Copy `.env.example` to `.env` and add any available API keys.
3. Run `python scripts/seed_mock_docs.py`.
4. Start the backend with `uvicorn app.main:app --reload --app-dir backend`.
5. Start the frontend with `streamlit run frontend/streamlit_app.py`.
6. Ask: "What should I know before using AI with enterprise data?"
7. Explain that the response is mocked while LangGraph, Pinecone, and LangSmith are placeholders.
