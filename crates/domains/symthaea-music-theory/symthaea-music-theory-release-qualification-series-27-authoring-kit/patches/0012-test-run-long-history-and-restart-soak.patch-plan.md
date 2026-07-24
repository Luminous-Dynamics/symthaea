# Patch 0012: test run long history and restart soak

**Series:** 27

## Objective

Exercise persistent lifecycle state over many publications, incidents, cycles, restarts, and audits.

## Intended changes

- Generate a deterministic long history with multiple segments, reopenings, recoveries, quarantines, and final retirement.
- Restart and reload between randomized but seeded operations.
- Audit full and incremental state repeatedly.

## Required tests

- No identity, ordinal, ledger, or active-state drift occurs.
- Incremental and full audit agree.
- Final archive reproduces from persisted state.

## Non-claims

- Does not model unbounded production scale.
- Does not replace real storage durability tests.
