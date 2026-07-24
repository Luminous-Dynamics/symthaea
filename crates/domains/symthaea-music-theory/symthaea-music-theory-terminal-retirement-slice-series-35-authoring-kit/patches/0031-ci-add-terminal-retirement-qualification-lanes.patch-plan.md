# Patch 0031: ci add terminal retirement qualification lanes

**Series:** 35

## Objective

Add focused build, test, conformance, transaction, privacy, archive, and reproduction lanes.

## Intended changes

- Run formatting, all-target/all-feature affected-crate checks and tests, Clippy, Nix, external verifier, race, rollback, mutation inventory, privacy, and deterministic archive lanes.
- Retain exact toolchain identities.
- Update the Series 30 backlog from observed evidence.

## Acceptance evidence

- No hard lane is silently skipped.
- Artifacts are emitted only after success.
- Clean replay reproduces the slice tree.

## Non-claims

- Does not claim these lanes ran in this kit.
- Does not qualify a successor system.
