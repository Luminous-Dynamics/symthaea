# Patch 0010: feat recovery add dual quorum cycle authorization

**Series:** 24

## Objective

Authorize one later-cycle recovery plan through active recovery authorities and recovered witnesses.

## Intended changes

- Define cycle-bound statements and authorization sets.
- Require exact active policy epochs at the intended recovery head.
- Report each quorum, signer exclusion, and external-verifier result independently.

## Required tests

- Duplicate, stale, quarantined, wrong-cycle, wrong-role, and externally rejected signers do not count.
- Prior-cycle authorization sets cannot be replayed.
- Threshold-edge valid authorization succeeds.

## Non-claims

- Does not prove signer independence.
- Does not make witness signatures publisher authority.
