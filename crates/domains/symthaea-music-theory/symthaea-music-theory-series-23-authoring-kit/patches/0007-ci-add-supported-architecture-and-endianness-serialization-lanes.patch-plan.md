# Patch 0007: Add supported architecture and serialization lanes

**Series:** 23

## Objective

Verify fixed-width persistence and canonical bytes across supported architectures.

## Intended changes

- Run native 64-bit and at least one 32-bit compile/test lane where toolchain support exists.
- Run canonical-vector checks on multiple architectures.
- Record unsupported architecture lanes as explicit unavailable evidence, not passes.

## Required tests

- Canonical digests are identical across exercised architectures.
- Platform-sized persistence is rejected.
- Integer-boundary fixtures produce identical failure codes.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
