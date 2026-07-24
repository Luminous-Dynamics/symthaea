# Patch 0013: test: conformance freeze fuzz seeds and property replay

## Objective

Retain minimized decoder/auditor counterexamples and replay them deterministically.

## Required evidence

- Frozen fixture IDs and expected outcomes.
- Independent recomputation of canonical bytes and identities.
- Stable failure stage rather than free-form error matching.
- Bounded parsers and verifier subprocess I/O.
- No private study, participant, credential, or governance secret data in public kits.
- Backward compatibility is explicit; silent expected-output edits are forbidden.
