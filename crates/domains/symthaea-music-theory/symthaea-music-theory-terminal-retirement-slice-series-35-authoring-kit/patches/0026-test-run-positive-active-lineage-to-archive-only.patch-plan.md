# Patch 0026: test run positive active lineage to archive only

**Series:** 35

## Objective

Prove the complete retirement slice through public APIs and CLI.

## Intended changes

- Begin with the qualified Series 34 active lineage.
- Evaluate trigger policy, authorize retirement, commit atomically, and verify archive-only mode and disclosure.
- Attempt every blocked mutation.

## Acceptance evidence

- API and CLI outputs agree.
- The exact terminal post-state is reached.
- The scenario reproduces byte-for-byte.

## Non-claims

- Does not prove keys were physically destroyed.
- Does not qualify a successor.
