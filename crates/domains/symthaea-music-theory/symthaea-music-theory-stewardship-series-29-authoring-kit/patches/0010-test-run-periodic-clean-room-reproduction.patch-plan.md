# Patch 0010: test run periodic clean room reproduction

**Series:** 29

## Objective

Detect environmental drift that breaks mail replay, builds, or release artifact reproducibility.

## Intended changes

- Recreate the release from source, patches, lockfiles, toolchain, and documented environment.
- Run offline or dependency-pinned modes where supported.
- Compare source tree, canonical vectors, and public archive identities.

## Required evidence

- Drift is classified as toolchain, dependency, environment, nondeterminism, or documentation failure.
- Failure creates a triage item.
- No differing artifact is published under the old identity.

## Non-claims

- Does not guarantee every external package remains downloadable forever.
- Does not redefine accepted evidence when reproduction fails.
