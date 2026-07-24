# Patch 0021: test add property fuzz and model checking seeds

**Series:** 26

## Objective

Find lifecycle inconsistencies beyond hand-authored fixtures.

## Intended changes

- Add property tests for append-only ledgers, ordinal continuity, one-active-state invariants, idempotency, and canonical identity stability.
- Fuzz decoders, canonicalizers, state transitions, and archive readers under fixed budgets.
- Use a small state-machine model to explore valid and invalid transition sequences.

## Required tests

- Every discovered failure becomes a frozen regression seed.
- Fuzzing cannot commit state outside a temporary store.
- Model and implementation agree for bounded state spaces.

## Non-claims

- Does not prove correctness for unbounded state spaces.
- Does not replace independent review.
