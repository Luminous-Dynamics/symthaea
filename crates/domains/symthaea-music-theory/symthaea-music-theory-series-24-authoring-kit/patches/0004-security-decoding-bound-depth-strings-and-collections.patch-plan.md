# Patch 0004: Bound decoding depth, strings, and collections

**Series:** 24

## Objective

Prevent nesting, string, map, and sequence bombs across public models.

## Intended changes

- Use depth-aware decoding or bounded intermediate representations.
- Reject duplicate keys and identities while building bounded collections.
- Avoid collecting unbounded iterators before count checks.

## Required tests

- Deeply nested artifacts fail at configured depth.
- Oversized strings fail before copying where possible.
- Duplicate-heavy inputs do not trigger quadratic behavior.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
