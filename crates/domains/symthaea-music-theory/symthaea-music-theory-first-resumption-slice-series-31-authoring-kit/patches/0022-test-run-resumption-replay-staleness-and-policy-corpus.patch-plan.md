# Patch 0022: test run resumption replay staleness and policy corpus

**Series:** 31

## Objective

Freeze the complete negative matrix for the first slice.

## Intended changes

- Cover old closure signature, old delegation, old allowance, stale head, changed policy, rotated authority, new quarantine, wrong channel, wrong segment, duplicate signer, and receipt replay.
- Run native and independent verification.
- Require stable stage and issue codes.

## Acceptance evidence

- No negative case mutates state or consumes allowance.
- Policy substitution never succeeds.
- Valid threshold-edge cases remain accepted.

## Non-claims

- Does not claim exhaustive key-compromise coverage.
- Does not replace fuzzing.
