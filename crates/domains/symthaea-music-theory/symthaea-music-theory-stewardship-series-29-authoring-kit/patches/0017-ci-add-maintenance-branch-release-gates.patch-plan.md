# Patch 0017: ci add maintenance branch release gates

**Series:** 29

## Objective

Apply the same hard invariants to patch releases and backports as to the initial qualified release.

## Intended changes

- Run formatting, all-target/all-feature checks, tests, Clippy, Nix, conformance, regression, compatibility, privacy, advisory, and deterministic archive gates.
- Allow narrower performance lanes only with documented equivalence.
- Reject unlinked fixes.

## Required evidence

- Every maintained branch has an explicit gate matrix.
- Skipped hard gates require a visible, authorized waiver.
- Artifacts are produced only after gates pass.

## Non-claims

- Does not guarantee CI infrastructure availability.
- Does not allow availability pressure to bypass semantic checks.
