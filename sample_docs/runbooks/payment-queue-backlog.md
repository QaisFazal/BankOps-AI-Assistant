---
department: payments
document_type: runbook
access_level: internal
created_date: 2025-02-11
title: Payment Queue Backlog Runbook
source_file: runbooks/payment-queue-backlog.md
---

# Payment Queue Backlog Runbook

## Purpose

Use this runbook when payment processing queues grow faster than workers can
clear them.

## Detection

Primary alerts include queue depth above 50,000 messages, worker error rate
above 2 percent, or oldest message age above 10 minutes.

## Steps

1. Identify whether backlog is isolated to instant payments, bill pay, ACH, or
   internal transfers.
2. Verify downstream systems are accepting requests.
3. Scale workers by 25 percent if downstream dependencies are healthy.
4. Pause non-critical batch ingestion if customer-facing traffic is impacted.
5. Monitor duplicate detection and ledger posting dashboards.

## Communications

Notify customer support if customer-facing payment delays exceed 15 minutes.
Notify treasury operations for ACH or corporate payment delays.
