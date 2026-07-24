# Patch 0007: Bound canonicalization and hashing bytes

**Series:** 24

## Objective

Prevent validly shaped but enormous objects from forcing unbounded canonical buffers.

## Intended changes

- Add counting sinks and streaming SHA-256 writers.
- Cap per-object and cumulative canonical bytes.
- Avoid materializing canonical bytes when only a digest is required, while retaining vector-export paths under limits.

## Required tests

- Digest-only verification stays within memory budget.
- Canonical-vector export fails cleanly above export limit.
- Within-limit canonical bytes remain exactly unchanged.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
