# Patch 0019: feat api export curated cycle recovery surface

**Series:** 24

## Objective

Expose cycle-aware recovery through a deliberate public API.

## Intended changes

- Export models, builders, auditors, verifier traits, transaction results, and issue codes.
- Preserve prior first-cycle APIs through compatible adapters where honest.
- Add compile-oriented first-cycle and second-cycle examples.

## Required tests

- Examples compile using only curated exports.
- No private type leaks into public signatures.
- Adapter behavior is byte-equivalent where compatibility is claimed.

## Non-claims

- Does not freeze internal module layout.
- Does not promise semver before declaration.
