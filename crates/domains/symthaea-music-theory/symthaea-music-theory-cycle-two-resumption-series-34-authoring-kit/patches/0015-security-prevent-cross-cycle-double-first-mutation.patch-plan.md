# Patch 0015: security prevent cross cycle double first mutation

**Series:** 34

## Objective

Ensure the new segment receives exactly one first mutation and the old frozen segment cannot restart.

## Intended changes

- Use expected-head and successor first-mutation slot preconditions.
- Reject mutation against the predecessor frozen segment.
- Define idempotent retry behavior.

## Acceptance evidence

- Two valid successor plans produce exactly one committed receipt.
- The loser consumes no allowance.
- Any predecessor-segment mutation fails.

## Non-claims

- Does not guarantee fairness.
- Does not solve distributed locking.
