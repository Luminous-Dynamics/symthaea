# Patch 0019: feat implement successor handoff with discontinuity

**Series:** 35

## Objective

Export evidence to a successor without inheriting the retired lineage's authority.

## Intended changes

- Bind source terminal checkpoint, object inventory, target identity, target schema, migration receipts, and explicit continuity claim.
- Default continuity to none.
- Require a new catalog and authority genesis for future publication.

## Acceptance evidence

- Same-identity restart, implicit continuity, target substitution, and missing source evidence fail.
- Import does not confer publication authority.
- Migration disagreement blocks continuity claims.

## Non-claims

- Does not endorse the successor.
- Does not create successor governance.
