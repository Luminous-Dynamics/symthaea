# Patch 0005: test: conformance cover order duplicates unknown fields and integer bounds

## Objective

Freeze collection-ordering, duplicate, schema, and fixed-width numeric behavior.

## Required evidence

- Frozen fixture IDs and expected outcomes.
- Independent recomputation of canonical bytes and identities.
- Stable failure stage rather than free-form error matching.
- Bounded parsers and verifier subprocess I/O.
- No private study, participant, credential, or governance secret data in public kits.
- Backward compatibility is explicit; silent expected-output edits are forbidden.
