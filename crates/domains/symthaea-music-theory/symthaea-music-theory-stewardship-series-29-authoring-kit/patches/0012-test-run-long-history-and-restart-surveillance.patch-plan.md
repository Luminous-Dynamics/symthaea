# Patch 0012: test run long history and restart surveillance

**Series:** 29

## Objective

Catch performance, persistence, and lifecycle errors that appear only after prolonged use.

## Intended changes

- Exercise long catalogs, many incidents, repeated recovery cycles, segment succession, retirement, restart, and archive-only verification.
- Track memory, latency, storage growth, and audit completion.
- Retain representative failing histories.

## Required evidence

- Threshold regressions create triage records.
- Restart does not alter derived lifecycle state.
- Global ordinals and ledger identities remain stable.

## Non-claims

- Does not predict all production workloads.
- Does not weaken limits to make soak tests pass.
