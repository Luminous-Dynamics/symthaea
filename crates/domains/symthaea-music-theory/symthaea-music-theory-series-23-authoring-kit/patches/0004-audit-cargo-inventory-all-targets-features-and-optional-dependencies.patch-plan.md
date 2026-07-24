# Patch 0004: Inventory all Cargo targets, features, and optional dependencies

**Series:** 23

## Objective

Make the actual build surface explicit before selecting CI lanes.

## Intended changes

- Derive library, bins, examples, tests, benches, build scripts, features, required-features, and optional dependency use from Cargo metadata.
- Reject binaries that import optional dependencies without required feature declarations.
- Reject source modules referenced by targets but omitted from the library module graph when they are intended public implementation.

## Required tests

- Freeze a machine-readable target matrix.
- Detect an orphan target or optional-dependency mismatch.
- Verify every excluded target has a checked reason and owner.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
