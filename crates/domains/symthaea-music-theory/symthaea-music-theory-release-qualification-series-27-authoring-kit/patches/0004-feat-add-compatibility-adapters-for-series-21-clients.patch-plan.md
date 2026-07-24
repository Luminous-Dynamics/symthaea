# Patch 0004: feat add compatibility adapters for series 21 clients

**Series:** 27

## Objective

Preserve honest backward verification and selected construction workflows for Series 21 users.

## Intended changes

- Add explicit adapters for compatible historical APIs and data roles.
- Return typed migration-required or unsupported errors where semantics changed.
- Keep original bytes and verification behavior available.

## Required tests

- Series 21 fixture corpus remains verifiable.
- Adapters cannot fabricate segments, cycles, or retirement state.
- Compatibility behavior is covered by snapshots.

## Non-claims

- Does not promise old mutation APIs remain supported.
- Does not silently reinterpret historical data.
