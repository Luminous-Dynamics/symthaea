# Patch 0005: feat recovery model cycle scoped authority epochs

**Series:** 24

## Objective

Make exceptional recovery authority activation explicit for each recovery generation.

## Intended changes

- Bind recovery-authority epochs to a cycle identity, activation checkpoint, predecessor authority epoch, and exact freeze lineage.
- Allow dual-quorum rotation into a new cycle without silently retaining authority from an abandoned cycle.
- Record outgoing, incoming, active, quarantined, and historical-only statuses.

## Required tests

- Wrong-cycle, pre-freeze, stale, unchanged, and disconnected authority epochs fail.
- Outgoing and incoming thresholds are independently checked.
- Historical-only authorities cannot sign new recovery decisions.

## Non-claims

- Does not define key custody.
- Does not require the same authority structure in every cycle.
