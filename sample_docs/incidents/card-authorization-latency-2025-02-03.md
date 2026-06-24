---
department: cards
document_type: incident
access_level: internal
created_date: 2025-02-03
title: Card Authorization Latency Incident
source_file: incidents/card-authorization-latency-2025-02-03.md
---

# Card Authorization Latency Incident

## Summary

Card authorization response times rose above the 2 second service objective
between 17:10 and 17:48 UTC. Point-of-sale transactions continued to process,
but some merchants retried authorizations.

## Impact

- Peak p95 authorization latency reached 6.8 seconds.
- Approximately 4,200 authorizations exceeded normal response targets.
- Fraud scoring continued operating in degraded mode.

## Root Cause

A new fraud model feature flag caused synchronous enrichment calls to the customer
risk service. The risk service was healthy, but not sized for card authorization
traffic patterns.

## Resolution

The feature flag was disabled, authorization services were restarted, and the
fraud team moved enrichment calls back to asynchronous evaluation.

## Lessons Learned

Latency budgets must be reviewed for every synchronous dependency in the card
authorization path.
