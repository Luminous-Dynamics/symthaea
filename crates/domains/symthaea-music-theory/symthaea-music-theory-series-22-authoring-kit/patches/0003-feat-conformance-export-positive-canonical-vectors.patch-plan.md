# Patch 0003: feat: conformance export positive canonical vectors

## Objective

Freeze canonical bytes, SHA-256 identities, and decoded summaries for representative valid models.

## Required evidence

- Frozen fixture IDs and expected outcomes.
- Independent recomputation of canonical bytes and identities.
- Stable failure stage rather than free-form error matching.
- Bounded parsers and verifier subprocess I/O.
- No private study, participant, credential, or governance secret data in public kits.
- Backward compatibility is explicit; silent expected-output edits are forbidden.
