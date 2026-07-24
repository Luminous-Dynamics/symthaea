# Patch 0009: feat recovery model cycle plan and limitations

**Series:** 24

## Objective

Define one exact later-cycle recovery decision without duplicating or weakening prior recovery contracts.

## Intended changes

- Add cycle-aware recovery policy and plan binding the frozen segment, cycle ledger, active authorities, witness policy, quarantines, candidate branch, expected advance, and mandatory limitations.
- Support stricter policy for repeated incidents and repeated signer compromise.
- Require verifier-owned expected policy identities.

## Required tests

- Wrong cycle, stale head, insufficient advance, missing quarantine, and policy mismatch fail.
- Every semantically relevant field changes the plan digest.
- A valid plan alone cannot mutate recovery state.

## Non-claims

- Does not guarantee successful re-entry.
- Does not treat repeat recovery as routine publication.
