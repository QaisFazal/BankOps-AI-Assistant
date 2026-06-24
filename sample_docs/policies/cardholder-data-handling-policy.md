---
department: information_security
document_type: policy
access_level: restricted
created_date: 2025-03-19
title: Cardholder Data Handling Policy
source_file: policies/cardholder-data-handling-policy.md
---

# Cardholder Data Handling Policy

## Purpose

This policy defines controls for systems and employees that handle cardholder
data.

## Rules

- Full card numbers must not be displayed in internal tools unless explicitly
  approved for a regulated operational process.
- Logs must never contain full magnetic stripe data, CVV values, or PIN data.
- Tokenization is required for analytics and assistant retrieval workflows.
- Access reviews are required quarterly for card operations systems.

## AI Assistant Restrictions

The assistant may explain procedures and summarize policy. It must not output
full card numbers, CVV values, PIN data, or instructions to bypass tokenization.
