# Patch 0006: feat implement cycle scoped recovery authority policy

**Series:** 33

## Objective

Bind exceptional recovery authority to cycle two and the exact frozen lineage.

## Intended changes

- Add cycle-specific authority epoch, activation checkpoint, predecessor epoch, signer roles, thresholds, and historical-only status.
- Require caller-owned expected policy.
- Reject automatic carryover of cycle-one authorization.

## Acceptance evidence

- Cycle-one, stale, quarantined, wrong-role, and disconnected authority statements do not count.
- Outgoing and incoming transition evidence is independently reported where rotation occurs.
- Policy substitution fails.

## Non-claims

- Does not manage keys.
- Does not prove organizational independence.
