# Patch 0015: Add fuzz, property, and complexity replay

**Series:** 24

## Objective

Discover and preserve new pathological inputs without relying only on hand-authored fixtures.

## Intended changes

- Fuzz decoders, canonicalization, lineage, witness sets, archive headers, and external-protocol messages.
- Assert no panic, uncontrolled allocation, path escape, accepted partial state, or nondeterministic result.
- Freeze minimized counterexamples with seed and tool version.

## Required tests

- All frozen seeds replay in normal CI.
- OOM-like cases are represented by bounded generators.
- Counterexample corrections require explicit corpus versioning.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
