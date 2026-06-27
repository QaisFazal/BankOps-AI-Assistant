# Observability

This project uses LangSmith to trace the assistant flow across the FastAPI chat
request, LangGraph workflow, tool calls, and retrieval operations.

## Enable Tracing

Copy `.env.example` to `.env` and fill in your LangSmith API key:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=bankops-ai-assistant
```

Tracing is only sent to LangSmith when both `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are configured. If the key is missing, the app keeps running
locally without remote traces.

## What Gets Traced

The backend creates these trace spans:

- `langgraph_assistant_run`: the top-level assistant run for one `/chat` request.
- `langgraph_assistant_stream`: the top-level assistant run for one `/chat/stream`
  request.
- `gemini_generate_answer`: the Gemini final-answer generation call.
- `knowledge_search_tool`: the server-side search tool call.
- `hybrid_retrieval`: the local or Pinecone-backed retrieval operation.
- `python_analysis_tool`: structured incident analytics tool execution.
- `dummy_mcp_tool`: dummy enterprise data tool execution.

Each traced run includes useful metadata:

- `user_id`
- `role`
- `session_id`
- tool or retriever details when relevant

This makes it easier to filter traces by user session, compare role behavior, and
debug why a response used a specific source.

## How To View Traces

1. Start the backend with your `.env` loaded:

   ```powershell
   uvicorn app.main:app --reload --app-dir backend
   ```

2. Send a chat request from the Streamlit frontend or with curl/Postman.

3. Open [LangSmith](https://smith.langchain.com/).

4. Select the `bankops-ai-assistant` project.

5. Open a `langgraph_assistant_run` trace.

6. Inspect child runs for tool calls and retrieval. Check metadata filters such
   as `user_id`, `role`, and `session_id` when debugging a specific conversation.

## Verify Gemini Generation

Use this checklist to prove the LLM path is active:

1. Confirm `.env` has:

   ```env
   GEMINI_API_KEY=your_gemini_key
   GEMINI_MODEL=gemini-1.5-flash
   LANGSMITH_TRACING=true
   ```

2. Restart the backend after editing `.env`.

3. Ask a retrieval-backed question:

   ```text
   Summarize payment outage incidents and cite the sources.
   ```

4. In the API response or Streamlit sidebar, check:

   ```text
   gemini generation completed
   ```

5. In LangSmith, open `langgraph_assistant_run` and confirm there is a child run:

   ```text
   gemini_generate_answer
   ```

6. Open `gemini_generate_answer` and inspect:

   - input question
   - context count
   - `user_id`, `role`, and `session_id` metadata
   - answer preview in outputs

7. Confirm the final answer cites chunk ids that also appear in the returned
   citations list.

If Gemini is unavailable, the trace or sidebar should show fallback behavior
instead of a stack trace:

```text
gemini fallback used
```

## Design Notes

The app uses a lightweight manual tracing wrapper instead of tracing every
function automatically. That keeps the beginner version easy to understand while
still showing the important production checkpoints:

- request enters the assistant graph
- retrieval or tool calls happen
- citations and result counts are attached to trace outputs
- session metadata follows the run across layers
