# Patch 0024: ci add cumulative all target feature and nix lanes

**Series:** 26

## Objective

Make the implementation continuously prove its complete target and feature surface.

## Intended changes

- Run formatting, minimal features, default features, all features, all targets, examples, doctests, Clippy, serialization fixtures, independent verifier, and Nix lanes.
- Detect orphan modules, unguarded optional dependencies, and binaries missing required features.
- Publish exact toolchain and dependency-lock identities.

## Required tests

- Every public binary and example compiles in its declared feature lane.
- All-target/all-feature tests and Clippy are clean.
- Clean Nix builds reproduce on two workspaces.

## Non-claims

- Does not claim cross-platform support not actually executed.
- Does not hide ignored or waived tests.
