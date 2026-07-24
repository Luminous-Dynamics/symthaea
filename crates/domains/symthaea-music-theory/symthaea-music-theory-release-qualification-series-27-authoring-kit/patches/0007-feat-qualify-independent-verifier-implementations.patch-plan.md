# Patch 0007: feat qualify independent verifier implementations

**Series:** 27

## Objective

Demonstrate that verification is not dependent on one Rust codebase.

## Intended changes

- Define qualification profiles for canonical decoding, signature request construction, policy checking, ledger audit, and lifecycle report validation.
- Run at least one independent implementation over the frozen corpus.
- Record implementation identity, version, toolchain, and unsupported roles.

## Required tests

- All stable-role results agree exactly or release is blocked.
- Unsupported roles are visible rather than guessed.
- Verifier policy is supplied externally in every accepted case.

## Non-claims

- Does not declare independent implementations bug-free.
- Does not use consensus among verifiers to resolve semantics.
