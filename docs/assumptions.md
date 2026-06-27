# Assumptions, Tradeoffs, and Known Limitations

## Assumptions

- The project is an assignment prototype, not a production banking system.
- All documents in `sample_docs/` are mock enterprise documents.
- Users are represented by request fields instead of a real identity provider.
- Roles are hardcoded as `viewer`, `analyst`, and `administrator`.
- The first retrieval store is local JSONL so the RAG behavior can be tested
  without paid infrastructure.
- Pinecone is optional and can be enabled by environment configuration.
- LangSmith is optional and only sends traces when an API key is configured.
- Gemini is the real LLM provider for final response generation.
- The deterministic answer builder remains as a fallback when Gemini credentials
  are missing or the provider fails.

## Design Tradeoffs

### Local First Retrieval

Local JSONL retrieval is easy to run, test, and explain. It avoids external
dependencies during development. The tradeoff is that it does not provide
production vector indexing, horizontal scale, or managed search operations.

### Gemini Embeddings With Local Fallback

Dense retrieval uses `gemini-embedding-2` with 768-dimensional vectors. Query
and document inputs use different retrieval prefixes recommended for asymmetric
search. Local retrieval falls back to same-size deterministic hash vectors when
Gemini is unavailable, keeping development usable at lower semantic quality.
Pinecone ingestion requires Gemini and never silently stores fallback vectors.

### Hardcoded RBAC

Hardcoded RBAC makes security behavior visible and testable for the assignment.
The tradeoff is that production systems should use identity-provider groups,
policy engines, audit records, and centralized access management.

### In-Memory Session Memory

Session memory is process-local and keyed by `session_id`. It is fast and simple,
but it is lost on restart and not shared across multiple backend instances.
Production memory should use durable encrypted storage with retention controls.

### Simplified RLM

The recursive planning node is deterministic rather than LLM-generated. It
demonstrates decomposition, depth limits, batch retrieval, and aggregation
without cost or nondeterminism. The tradeoff is that plans are less flexible than
real model-generated plans.

### Gemini With Deterministic Fallback

Gemini provides the final user-facing answer when credentials are configured.
The fallback builder keeps the demo reliable when the API key is missing, the
provider times out, or the SDK is unavailable. The tradeoff is that local runs
without a Gemini key still produce template-style answers.

## Known Limitations

- No real enterprise authentication or SSO.
- No durable user/session memory.
- No production document ingestion pipeline for PDFs, Word files, permissions,
  or incremental updates.
- Pinecone ingestion supports JSONL chunk upserts, but it is a manual batch
  script rather than an incremental production ingestion service.
- Local retrieval recomputes document embeddings for each search instead of
  caching them.
- Gemini embedding availability, quotas, latency, and cost affect semantic
  retrieval.
- Streaming emits live graph activity, then splits the completed Gemini answer
  into token events rather than using provider-native token streaming.
- The analytics tool is implemented and tested but is not yet invoked by the
  LangGraph workflow.
- The dummy MCP-style tool is not a real MCP server.
- Citation validation reports invalid citations but does not yet replace the
  generated answer with a blocked response.
- No human feedback loop or evaluation dataset.
- No tenant isolation beyond mock role and metadata filtering.
- No production secrets management.
- No deployment hardening, TLS termination, or container security policy.
- Prompt-injection protection is pattern-based and should be expanded with
  layered defenses before production use.

## Security Notes

- `.env.example` must not contain real secrets.
- `.env` should stay local and be ignored by git.
- Every tool checks permissions server-side.
- Viewer users can only search `internal` documents.
- Analyst users can access `internal` and `confidential` documents.
- Administrator users can access all mock access levels.

## Future Improvements

- Cache local document embeddings and batch large ingestion workloads.
- Extend Pinecone ingestion with incremental updates, deletion handling, and
  source permission synchronization.
- Route analytics requests through `python_analysis_tool` in LangGraph.
- Replace the dummy MCP-style tool with a real MCP server and client.
- Replace answers when citation validation fails.
- Replace hardcoded RBAC with enterprise identity and policy checks.
- Store memory in Redis or Postgres with encryption and retention settings.
- Add eval traces and feedback scoring in LangSmith.
- Add richer guardrails for data-loss prevention, prompt injection, and output
  policy validation.
- Add document-level permission sync from source systems.
