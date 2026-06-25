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
- The current answer generator is deterministic and template-based. It stands in
  for a future LLM call.

## Design Tradeoffs

### Local First Retrieval

Local JSONL retrieval is easy to run, test, and explain. It avoids external
dependencies during development. The tradeoff is that it does not provide
production vector indexing, horizontal scale, or managed search operations.

### Hash Embeddings

The dense embedding provider uses deterministic hash embeddings. This keeps
tests repeatable and avoids requiring OpenAI credentials. The tradeoff is that
semantic quality is much weaker than real embeddings.

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

### Deterministic Answer Generation

The assistant currently formats retrieved evidence into readable answers instead
of calling a real LLM. This makes citation validation easier and prevents
hallucinated language during the assignment. The tradeoff is that answers are
less conversational and less capable of synthesis.

## Known Limitations

- No real enterprise authentication or SSO.
- No durable user/session memory.
- No production document ingestion pipeline for PDFs, Word files, permissions,
  or incremental updates.
- No Pinecone upsert script yet.
- No real OpenAI embedding or chat model call wired into the graph.
- No streaming responses.
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

- Add OpenAI embeddings and chat model integration behind abstractions.
- Add Pinecone ingestion/upsert support.
- Replace hardcoded RBAC with enterprise identity and policy checks.
- Store memory in Redis or Postgres with encryption and retention settings.
- Add eval traces and feedback scoring in LangSmith.
- Add richer guardrails for data-loss prevention, prompt injection, and output
  policy validation.
- Add document-level permission sync from source systems.
