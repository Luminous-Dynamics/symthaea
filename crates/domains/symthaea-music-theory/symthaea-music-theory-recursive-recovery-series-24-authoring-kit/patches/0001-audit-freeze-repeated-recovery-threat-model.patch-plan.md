# Patch 0001: audit freeze repeated recovery threat model

**Series:** 24

## Objective

Freeze the failure modes introduced when a previously recovered and resumed system is reopened and must recover again.

## Intended changes

- Inventory all Series 20–23 incident, recovery, re-entry, closure, segment, resumption, challenge, and freeze objects that could be confused across cycles.
- Model cross-cycle signature replay, branch substitution, stale quarantine release, ordinal reset, generation skipping, cycle collapse, and concurrent recovery attempts.
- Define the exact immutable predecessor facts required to begin a later recovery cycle.

## Required tests

- Every state-changing surface is assigned to one recovery cycle.
- The audit distinguishes incident identity, recovery-cycle identity, trust-segment identity, authority epoch, and catalog head.
- Each threat maps to a required structural, authenticated-policy, or transaction failure.

## Non-claims

- Does not claim every future incident is recoverable.
- Does not permit a later cycle to rewrite the first recovery.
