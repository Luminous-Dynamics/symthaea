# Patch 0025: ci add first slice build test and reproduction lanes

**Series:** 31

## Objective

Create the smallest complete CI qualification lane for the vertical slice.

## Intended changes

- Run formatting, all-target/all-feature check and tests for affected crates, Clippy, Nix, external-verifier fixtures, race tests, and deterministic archives.
- Separate fast and full lanes.
- Retain exact toolchain identities.

## Acceptance evidence

- No hard lane is silently skipped.
- Artifacts are emitted only after success.
- Clean replay reproduces the slice tree.

## Non-claims

- Does not claim CI has run in this authoring kit.
- Does not qualify later lifecycle modules.
