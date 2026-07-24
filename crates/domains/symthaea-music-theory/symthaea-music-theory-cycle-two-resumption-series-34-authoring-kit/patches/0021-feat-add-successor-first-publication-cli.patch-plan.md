# Patch 0021: feat add successor first publication cli

**Series:** 34

## Objective

Expose dry-run and transactional commit for the first publication of the successor segment.

## Intended changes

- Require the accepted resumption package and exact mutable state.
- Reauthenticate at commit time.
- Write post-state and receipt atomically with overwrite protection.

## Acceptance evidence

- Dry-run is non-mutating.
- Interrupted output cannot appear accepted.
- Stale-state and race cases fail.

## Non-claims

- Does not publish to a remote service.
- Does not contact signers.
