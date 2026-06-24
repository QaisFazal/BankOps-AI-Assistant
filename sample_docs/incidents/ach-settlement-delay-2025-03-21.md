---
department: payments
document_type: incident
access_level: confidential
created_date: 2025-03-21
title: ACH Settlement Delay Incident
source_file: incidents/ach-settlement-delay-2025-03-21.md
---

# ACH Settlement Delay Incident

## Summary

The outbound ACH settlement file missed the 18:00 UTC submission window after a
batch validation job paused on an unexpected counterparty routing number format.

## Impact

- 37 corporate payroll batches were delayed to the next available window.
- Treasury operations manually notified relationship managers for affected
  corporate customers.
- No funds were lost and no ledger imbalance was detected.

## Root Cause

The validator rejected a valid nine-digit routing number because a new
counterparty profile lacked the expected institution alias.

## Resolution

Operations approved a controlled override, the missing alias was added, and the
file was submitted in the 20:00 UTC window.

## Preventive Controls

- Add counterparty profile validation to onboarding.
- Create a playbook for controlled ACH validation overrides.
- Add automated notification when settlement is projected to miss a cut-off.
