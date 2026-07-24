# Patch 0003: Add preflight measurement and early rejection

**Series:** 24

## Objective

Reject obviously oversized inputs before full decoding or expensive cryptography.

## Intended changes

- Measure raw bytes and safe envelope metadata first.
- Validate declared lengths against remaining bytes and local limits.
- Report exact limit dimension, observed value, configured maximum, and stage.

## Required tests

- Oversized raw input allocates only bounded memory.
- Forged huge declared lengths fail without allocation.
- The same input and policy yield the same first failure.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
