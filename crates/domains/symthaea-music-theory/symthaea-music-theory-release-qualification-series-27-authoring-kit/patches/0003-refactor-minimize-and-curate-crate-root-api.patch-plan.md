# Patch 0003: refactor minimize and curate crate root api

**Series:** 27

## Objective

Reduce accidental public surface before declaring compatibility.

## Intended changes

- Export only supported models, policies, verifier traits, transactions, reports, and stable issue codes.
- Move canonicalization and mutation internals behind private modules.
- Provide task-oriented prelude modules only where they reduce misuse.

## Required tests

- Public API snapshot shrinks or is explicitly justified.
- Examples compile using curated exports only.
- No private type leaks through public signatures.

## Non-claims

- Does not preserve accidental module paths.
- Does not hide required low-level verification controls.
