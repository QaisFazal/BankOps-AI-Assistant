---
department: enterprise_architecture
document_type: architecture
access_level: internal
created_date: 2025-02-15
title: Card Processing Reference Architecture
source_file: architecture/card-processing-reference.md
---

# Card Processing Reference Architecture

## Overview

Card processing supports authorization, clearing, settlement, disputes, fraud
signals, and cardholder notifications.

## Authorization Path

The low-latency path includes merchant network ingress, token resolution, account
status checks, available balance lookup, fraud scoring, decisioning, and response
publication.

## Data Stores

- Token vault stores card token mappings.
- Authorization ledger stores pending holds.
- Event archive stores authorization decisions and reversal events.
- Feature store supplies fraud and risk features.

## Non-Functional Requirements

- p95 authorization response time below 2 seconds.
- Regional failover within 10 minutes.
- Full audit history for authorization decisions.
