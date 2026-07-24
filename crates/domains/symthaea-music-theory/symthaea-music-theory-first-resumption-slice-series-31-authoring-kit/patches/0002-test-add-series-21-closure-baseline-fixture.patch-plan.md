# Patch 0002: test add series 21 closure baseline fixture

**Series:** 31

## Objective

Create the exact valid baseline fixture that the vertical slice consumes.

## Intended changes

- Package a structurally valid closure lineage with exact tree/archive provenance and synthetic cryptographic test identities.
- Keep private production material out of fixtures.
- Include malformed and stale baseline variants.

## Acceptance evidence

- Baseline passes native Series 21 verification.
- Mutated variants fail at expected stages.
- Fixture archive is deterministic.

## Non-claims

- Does not claim production signer identity.
- Does not prove the historical uploaded archive compiled.
