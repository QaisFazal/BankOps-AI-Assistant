---
department: cards
document_type: runbook
access_level: internal
created_date: 2025-01-20
title: Card Authorization Service Restart Runbook
source_file: runbooks/card-auth-service-restart.md
---

# Card Authorization Service Restart Runbook

## Purpose

Use this runbook when card authorization pods are unhealthy, latency exceeds the
service objective, or a controlled deployment rollback is required.

## Preconditions

- Confirm the incident commander has approved the restart.
- Confirm at least one authorization region remains healthy.
- Notify fraud operations before restarting services.

## Steps

1. Check current health dashboard for `card-auth-api`.
2. Drain one pod group at a time from the load balancer.
3. Restart pods using the approved deployment pipeline.
4. Wait for p95 latency to return below 2 seconds.
5. Confirm authorization approval and decline rates match historical baselines.

## Rollback

If latency increases after restart, roll back to the last stable deployment and
page the cards platform lead.
