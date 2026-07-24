# Patch 0025: feat add terminal retirement and archive cli

**Series:** 35

## Objective

Expose dry-run, transactional commit, archive verification, disclosure export, and successor handoff.

## Intended changes

- Require the accepted retirement package and exact mutable state.
- Reauthenticate at commit time.
- Write terminal state and packages atomically with overwrite protection.

## Acceptance evidence

- Dry-run is non-mutating.
- Interrupted writes leave no accepted partial retirement.
- Archive and handoff verification remain available after commit.

## Non-claims

- Does not destroy keys automatically.
- Does not contact a successor.
