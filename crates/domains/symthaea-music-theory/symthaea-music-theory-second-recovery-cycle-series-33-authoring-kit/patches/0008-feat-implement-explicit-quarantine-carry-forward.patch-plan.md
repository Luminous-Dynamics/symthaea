# Patch 0008: feat implement explicit quarantine carry forward

**Series:** 33

## Objective

Carry all unresolved Series 32 quarantines into cycle two unless separately authorized.

## Intended changes

- Snapshot witness, observer, authority, verifier, publisher, and artifact quarantines at cycle genesis.
- Represent carry-forward, escalation, replacement, and release actions.
- Require exact evidence and expected policy for release.

## Acceptance evidence

- Omitted quarantines remain active.
- Release for one identity or role cannot release another.
- Prior quarantine history remains immutable.

## Non-claims

- Does not assign blame.
- Does not make every quarantine permanent.
