---
department: product
document_type: product_spec
access_level: internal
created_date: 2025-02-27
title: Commercial Card Controls Product Specification
source_file: products/commercial-card-controls-spec.md
---

# Commercial Card Controls Product Specification

## Overview

Commercial Card Controls lets business administrators define spending controls
for employee cards.

## Capabilities

- Merchant category restrictions.
- Daily and monthly spend limits.
- Travel window controls.
- Real-time alerts for declined transactions.
- Bulk policy assignment by department.

## Operational Notes

Card control updates should propagate to authorization systems within five
minutes. Failed propagation must trigger an operations alert and prevent the UI
from showing the policy as active.
