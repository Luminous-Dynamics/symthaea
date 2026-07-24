# Patch 0011: feat add compare and commit state store contract

**Series:** 26

## Objective

Provide one atomic storage boundary for first publication, freeze, recovery, reopening, and retirement.

## Intended changes

- Define immutable snapshots, expected-head compare, staged writes, commit receipts, rollback semantics, and idempotency behavior.
- Supply an in-memory reference implementation and integration trait for durable stores.
- Require all affected ledgers and allowance counters to commit together.

## Required tests

- Failure injection at every stage leaves byte-identical pre-state.
- Two conflicting transitions from one head cannot both commit.
- Committed state passes all underlying audits.

## Non-claims

- Does not provide distributed consensus.
- Does not guarantee durability beyond a store implementation contract.
