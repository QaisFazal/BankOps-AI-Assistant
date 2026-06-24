---
department: cards
document_type: runbook
access_level: restricted
created_date: 2025-03-05
title: Card Network Failover Runbook
source_file: runbooks/card-network-failover.md
---

# Card Network Failover Runbook

## Purpose

This runbook describes controlled failover from the primary card network link to
the secondary link during network instability or scheduled maintenance.

## Authorization

Failover requires approval from the incident commander and the cards operations
manager. Fraud operations must be informed before traffic moves.

## Steps

1. Confirm the secondary link has passed connectivity checks.
2. Lower traffic weight on the primary link to 50 percent for five minutes.
3. Validate authorization response codes and settlement sequence numbers.
4. Move traffic to the secondary link if metrics remain stable.
5. Keep the primary link in standby until the network provider confirms recovery.

## Recovery

Return traffic gradually, starting at 25 percent, and monitor reversal rates for
at least 30 minutes.
