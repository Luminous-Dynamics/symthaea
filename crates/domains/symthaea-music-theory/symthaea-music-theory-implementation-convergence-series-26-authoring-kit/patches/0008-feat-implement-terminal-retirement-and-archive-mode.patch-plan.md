# Patch 0008: feat implement terminal retirement and archive mode

**Series:** 26

## Objective

Convert Series 25 retirement plans into a one-way compiled terminal state and read-only archive surface.

## Intended changes

- Implement trust-exhaustion policy and report, retirement plan, multi-role authorization, committed receipt, terminal capability revocations, terminal checkpoint, archive-only profile, custody records, successor handoff, and disclosure package.
- Require a new catalog identity for any successor publication.
- Keep historical verification available after mutation authority ends.

## Required tests

- Every mutation path fails after committed retirement.
- Archive verification remains functional without signing capability.
- Same-identity successor and implicit continuity are rejected.

## Non-claims

- Does not prove physical key destruction.
- Does not guarantee permanent storage.
