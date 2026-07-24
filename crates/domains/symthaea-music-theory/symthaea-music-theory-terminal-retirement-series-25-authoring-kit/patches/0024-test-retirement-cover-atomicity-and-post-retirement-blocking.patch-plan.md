# Patch 0024: test retirement cover atomicity and post retirement blocking

**Series:** 25

## Objective

Prove that retirement is one-way for the catalog identity and leaves no mutation capability active.

## Intended changes

- Inject failure at each transaction stage.
- Race retirement against publication, recovery, reopening, resumption, allowance issuance, and authority rotation.
- Attempt every mutation path after successful retirement.

## Required tests

- Zero or one conflicting transition commits.
- Failed retirement leaves byte-identical pre-state.
- All post-retirement mutation attempts fail under stable retired-lineage codes.

## Non-claims

- Does not control unmodeled external systems.
- Does not benchmark distributed contention.
