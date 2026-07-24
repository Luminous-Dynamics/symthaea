# Patch 0005: feat implement resumption policy authorization and receipts

**Series:** 26

## Objective

Implement the full Series 22 authorization model without deferring mutable checks to documentation.

## Intended changes

- Add verifier-owned resumption policy, canonical plan, limitations, dual-quorum statements, authorization set, fresh publisher delegation, fresh allowance binding, and first-mutation receipt.
- Require exact expected policy identities at verification.
- Keep authorization, commit eligibility, and successful mutation as separate results.

## Required tests

- Replay, stale-policy, wrong-segment, old-delegation, and old-allowance vectors fail.
- Threshold-edge valid authorization succeeds.
- Canonical bytes match frozen vectors.

## Non-claims

- Does not mutate the catalog.
- Does not reuse closure signatures as resumption signatures.
