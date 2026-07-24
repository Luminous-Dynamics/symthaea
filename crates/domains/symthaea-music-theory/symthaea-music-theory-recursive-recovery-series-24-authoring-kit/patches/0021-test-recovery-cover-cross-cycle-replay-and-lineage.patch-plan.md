# Patch 0021: test recovery cover cross cycle replay and lineage

**Series:** 24

## Objective

Freeze replay and lineage-confusion attacks across at least three cycles.

## Intended changes

- Cover old signatures, old witness sets, old quarantines, wrong predecessor, skipped cycle ordinal, reused segment, and candidate from another incident.
- Require stable issue codes and earliest failure stages.
- Run Rust and independent-verifier fixtures.

## Required tests

- No prior-cycle authority satisfies a later cycle.
- No cycle can disappear from the ledger.
- Valid first, second, and third cycle examples succeed.

## Non-claims

- Does not claim unlimited-cycle practicality.
- Does not prove signer compromise is impossible.
