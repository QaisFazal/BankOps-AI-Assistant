---
department: payments
document_type: incident
access_level: internal
created_date: 2025-01-14
title: Payment Gateway Timeout Incident
source_file: incidents/payment-gateway-timeout-2025-01-14.md
---

# Payment Gateway Timeout Incident

## Summary

At 09:42 UTC, debit transfers routed through the primary payment gateway began
timing out for mobile and branch channels. The incident affected domestic
instant payments and delayed roughly 18 percent of payment submissions.

## Customer Impact

- Customers saw pending transactions for up to 22 minutes.
- No duplicate posting was confirmed after reconciliation.
- Branch tellers used manual pending-payment guidance during the event.

## Root Cause

The gateway connection pool exhausted available worker threads after a vendor
certificate refresh increased TLS handshake latency.

## Resolution

The payments platform team increased the gateway worker pool, restarted the
affected pods, and confirmed settlement files matched the ledger by 11:05 UTC.

## Follow-Up Actions

- Add certificate refresh simulation to the monthly resiliency drill.
- Alert when handshake latency exceeds 500 ms for five minutes.
- Review gateway pool sizing before peak payroll dates.
