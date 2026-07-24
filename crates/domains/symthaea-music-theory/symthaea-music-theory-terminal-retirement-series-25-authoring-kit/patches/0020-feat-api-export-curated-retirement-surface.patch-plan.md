# Patch 0020: feat api export curated retirement surface

**Series:** 25

## Objective

Expose terminal-retirement and archive-only workflows deliberately.

## Intended changes

- Export models, builders, auditors, verifier traits, transaction outcomes, and stable issue codes.
- Keep mutation helpers inaccessible once the caller holds a retired-lineage token.
- Add compile-oriented retirement and archive examples.

## Required tests

- Examples compile using curated exports only.
- No private implementation type leaks into public signatures.
- The retired-lineage API cannot construct mutation commands.

## Non-claims

- Does not provide language-level guarantees against old binaries.
- Does not freeze internal module paths.
