---
department: information_security
document_type: policy
access_level: internal
created_date: 2025-01-07
title: AI Assistant Security Policy
source_file: policies/ai-assistant-security-policy.md
---

# AI Assistant Security Policy

## Purpose

This policy defines security expectations for AI assistants used by bank
employees and contractors.

## Requirements

- The assistant must authenticate users before answering enterprise questions.
- Responses must respect document access levels and user roles.
- The assistant must not reveal secrets, passwords, private keys, or customer
  identifiers.
- Prompt injection attempts must be logged and handled as suspicious input.
- Human approval is required for high-impact financial or operational decisions.

## Logging

Prompts, retrieved document identifiers, model responses, and tool actions must
be logged to an approved audit system.
