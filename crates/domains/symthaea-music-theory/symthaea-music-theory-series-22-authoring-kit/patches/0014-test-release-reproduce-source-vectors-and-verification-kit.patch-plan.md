# Patch 0014: test: release reproduce source vectors and verification kit

## Objective

Rebuild all distributed artifacts twice and require byte-identical outputs and verifier agreement.

## Required evidence

- Frozen fixture IDs and expected outcomes.
- Independent recomputation of canonical bytes and identities.
- Stable failure stage rather than free-form error matching.
- Bounded parsers and verifier subprocess I/O.
- No private study, participant, credential, or governance secret data in public kits.
- Backward compatibility is explicit; silent expected-output edits are forbidden.
