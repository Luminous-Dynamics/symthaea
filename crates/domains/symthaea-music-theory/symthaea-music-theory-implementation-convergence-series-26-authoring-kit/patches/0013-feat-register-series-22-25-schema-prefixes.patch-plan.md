# Patch 0013: feat register series 22 25 schema prefixes

**Series:** 26

## Objective

Implement one append-only stable schema registry for every new persisted role.

## Intended changes

- Assign numeric roles, versions, canonical encodings, unknown-field rules, and compatibility boundaries.
- Preserve every Series 21 role and byte vector unchanged.
- Generate registry documentation and independent fixtures from code.

## Required tests

- Role collisions, renumbering, debug-derived identities, and `usize` persistence fail CI.
- Rust and independent decoders agree on positive and negative vectors.
- Registry generation is deterministic.

## Non-claims

- Does not reserve unimplemented speculative roles.
- Does not make registration authority.
