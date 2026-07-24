# Patch 0007: feat: tools run no shell differential verifiers

## Objective

Invoke configured verifier programs directly with bounded JSON input/output and deterministic result comparison.

## Required evidence

- Frozen fixture IDs and expected outcomes.
- Independent recomputation of canonical bytes and identities.
- Stable failure stage rather than free-form error matching.
- Bounded parsers and verifier subprocess I/O.
- No private study, participant, credential, or governance secret data in public kits.
- Backward compatibility is explicit; silent expected-output edits are forbidden.
