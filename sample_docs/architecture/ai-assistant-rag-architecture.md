---
department: data_platform
document_type: architecture
access_level: internal
created_date: 2025-03-12
title: AI Assistant Retrieval Architecture
source_file: architecture/ai-assistant-rag-architecture.md
---

# AI Assistant Retrieval Architecture

## Overview

The AI assistant uses retrieval-augmented generation to answer employee
questions from approved enterprise documents. The system retrieves relevant
chunks, sends them to the model with task instructions, and returns citations.

## Ingestion

Documents are loaded from approved repositories, normalized to markdown, tagged
with metadata, chunked, embedded, and stored in a vector index.

## Query Flow

1. Validate user role and request.
2. Apply metadata filters based on access level.
3. Retrieve candidate chunks from the vector index.
4. Rerank results and build the answer prompt.
5. Return the answer with citations and trace details.

## Controls

The assistant must refuse answers that require restricted documents beyond the
user's role. All queries and citations are logged for audit review.
