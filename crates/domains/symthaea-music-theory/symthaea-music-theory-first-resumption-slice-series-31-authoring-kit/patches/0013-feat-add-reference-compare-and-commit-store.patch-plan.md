# Patch 0013: feat add reference compare and commit store

**Series:** 31

## Objective

Provide the transactional state boundary required by the slice.

## Intended changes

- Model exact expected head, staged catalog append, allowance consumption, segment update, and receipt publication.
- Support failure injection and restart inspection.
- Return typed conflict and rollback outcomes.

## Acceptance evidence

- Failure at every stage leaves byte-identical pre-state.
- Two commits from one head cannot both succeed.
- Restart yields one unambiguous committed or uncommitted result.

## Non-claims

- Does not implement distributed locking.
- Does not select a production storage engine.
