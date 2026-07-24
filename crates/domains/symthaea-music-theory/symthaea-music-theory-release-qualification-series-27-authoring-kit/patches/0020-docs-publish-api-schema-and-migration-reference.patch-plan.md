# Patch 0020: docs publish api schema and migration reference

**Series:** 27

## Objective

Provide one generated reference for stable APIs, schemas, compatibility adapters, and retirement behavior.

## Intended changes

- Document stability tiers, canonical roles, expected-policy inputs, state transitions, examples, compatibility windows, and unsupported operations.
- Generate from source inventory and fixtures.
- Include exact deprecation and removal guidance.

## Required tests

- Reference generation is deterministic.
- Undocumented stable exports fail CI.
- Historical roles remain discoverable.

## Non-claims

- Does not promise support beyond stated windows.
- Does not hand-edit implementation status.
