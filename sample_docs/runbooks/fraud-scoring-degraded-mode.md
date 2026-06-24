---
department: fraud_risk
document_type: runbook
access_level: confidential
created_date: 2025-04-18
title: Fraud Scoring Degraded Mode Runbook
source_file: runbooks/fraud-scoring-degraded-mode.md
---

# Fraud Scoring Degraded Mode Runbook

## Purpose

Use degraded mode when the fraud scoring service is unavailable or scoring
latency threatens customer transaction completion.

## Degraded Behavior

The authorization path uses cached customer risk bands and rule-based limits.
High-risk transactions are routed for enhanced review after authorization where
allowed by policy.

## Steps

1. Confirm scoring service health and dependency status.
2. Enable `fraud.degraded_mode` through the controlled feature flag workflow.
3. Confirm card authorization latency is within target.
4. Notify fraud operations and customer support.
5. Review sampled transactions every 15 minutes.

## Exit Criteria

Disable degraded mode only after scoring service p95 latency is below 300 ms for
30 consecutive minutes.
