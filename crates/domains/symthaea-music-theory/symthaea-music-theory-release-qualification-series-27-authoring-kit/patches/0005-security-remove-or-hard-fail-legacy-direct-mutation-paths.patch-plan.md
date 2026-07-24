# Patch 0005: security remove or hard fail legacy direct mutation paths

**Series:** 27

## Objective

Complete the transition away from APIs that bypass lifecycle gates.

## Intended changes

- Remove private-only helpers where safe and convert remaining compatibility shims to hard-fail with guidance.
- Block legacy CLI flags, routes, and persisted jobs from mutating authoritative state.
- Generate endpoint and command inventories in CI.

## Required tests

- Every legacy bypass attempt fails under a stable code.
- Read-only historical verification still works.
- No hidden route or feature restores direct mutation.

## Non-claims

- Does not control third-party forks.
- Does not remove archived documentation required for historical interpretation.
