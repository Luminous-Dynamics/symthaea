# Patch 0006: Add minimal, default, and all-feature Cargo matrix

**Series:** 23

## Objective

Exercise feature unification and optional code paths rather than relying on one maximal build.

## Intended changes

- Run format, check, test, and Clippy over defined feature profiles.
- Include doctests, examples, and binaries.
- Use locked dependency resolution and fail on lockfile mutation.

## Required tests

- Minimal and default lanes do not rely on all-feature side effects.
- All-feature lane catches incompatible optional paths.
- Warnings fail every release lane.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
