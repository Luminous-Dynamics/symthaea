# Patch 0013: feat retirement model successor handoff with explicit discontinuity

**Series:** 25

## Objective

Allow evidence export to a successor system without silently claiming continuous authority.

## Intended changes

- Define a handoff package binding retired source identity, terminal checkpoint, exported object inventory, target schema or catalog identity, migration receipts, and explicit continuity claim.
- Require continuity to default to `none` unless separately proven.
- Preserve source bytes and source verification instructions.

## Required tests

- Target substitution, omitted source evidence, and implicit continuity fail.
- A successor can import evidence without inheriting publication authority.
- Migration disagreement blocks continuity claims.

## Non-claims

- Does not endorse the successor.
- Does not authorize the successor to speak for the retired catalog.
