# Patch 0014: feat recovery model cycle closure

**Series:** 24

## Objective

Close a later recovery cycle without pretending the repeated incident never happened.

## Intended changes

- Add cycle-aware closure policy, plan, statements, dual-quorum authorization, bundle, and audit.
- Bind the accepted re-entry certification, all cycle quarantines, recurrence lineage, and required limitations.
- Permit stricter thresholds or mandatory independent review after repeated incidents.

## Required tests

- Wrong cycle, stale checkpoint, active forbidden quarantine, and reused earlier closure signatures fail.
- Closure remains a distinct decision after re-entry.
- The cycle ledger appends closure rather than rewriting state.

## Non-claims

- Does not resume publication by itself.
- Does not erase the reopened incident.
