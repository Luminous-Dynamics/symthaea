# Patch 0019: feat schema register retirement and archive contracts

**Series:** 25

## Objective

Append stable roles for trust exhaustion, terminal retirement, archive-only verification, and successor handoff.

## Intended changes

- Register policy, trigger report, plan, statements, authorization set, receipt, terminal checkpoint, archive profile, custody ledger, handoff package, observer statement, and disclosure package.
- Use fixed-width fields and stable numeric roles.
- Publish compatibility and unknown-field rules.

## Required tests

- Prior schema prefixes remain unchanged.
- Role collisions, `usize` persistence, and debug-derived encodings fail CI.
- Independent fixtures decode or reject identically.

## Non-claims

- Does not reserve speculative resurrection roles.
- Does not make schema registration authority.
