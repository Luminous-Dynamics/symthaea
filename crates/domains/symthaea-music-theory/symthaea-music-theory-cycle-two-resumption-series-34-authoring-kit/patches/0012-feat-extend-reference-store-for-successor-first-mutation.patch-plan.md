# Patch 0012: feat extend reference store for successor first mutation

**Series:** 34

## Objective

Add transactional successor-segment activation and first publication.

## Intended changes

- Stage segment activation, catalog append, status event, allowance consumption, global ordinal advances, first-mutation reference, and receipt.
- Use compare-and-commit against exact cycle, segment, and catalog heads.
- Support failure injection and restart.

## Acceptance evidence

- Failure at each stage leaves byte-identical pre-state.
- Two first-mutation attempts cannot both commit.
- Restart yields one unambiguous state.

## Non-claims

- Does not implement distributed consensus.
- Does not choose a production database.
