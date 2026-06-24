---
department: digital_banking
document_type: incident
access_level: internal
created_date: 2025-04-09
title: Mobile Bill Pay Degradation
source_file: incidents/mobile-bill-pay-degradation-2025-04-09.md
---

# Mobile Bill Pay Degradation

## Summary

Mobile bill pay requests returned intermittent 503 errors during a planned
database maintenance period. The web channel was unaffected.

## Impact

- 9 percent of mobile bill pay attempts failed for 31 minutes.
- Failed attempts were not submitted to billers.
- Customer support volume increased by 14 percent during the incident.

## Root Cause

The mobile bill pay API used a read replica that was removed from rotation
earlier than expected by the database maintenance workflow.

## Resolution

The API was switched to the secondary replica pool and the maintenance workflow
was updated to drain replicas only after health checks confirm replacement
capacity.

## Follow-Up Actions

- Add mobile-specific synthetic tests for bill pay.
- Require application owner sign-off for replica removal.
