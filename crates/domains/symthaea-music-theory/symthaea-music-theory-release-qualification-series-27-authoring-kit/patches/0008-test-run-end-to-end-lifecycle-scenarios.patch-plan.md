# Patch 0008: test run end to end lifecycle scenarios

**Series:** 27

## Objective

Exercise complete state histories through public APIs and commands, not internal helpers.

## Intended changes

- Run normal closure-to-resumption, challenge-to-freeze, repeated recovery, successful retirement, archive verification, and successor-discontinuity scenarios.
- Persist and reload state between each major phase.
- Verify all public packages independently.

## Required tests

- Every scenario produces exact expected heads, ordinals, ledgers, reports, and receipts.
- Restart and reload do not alter identities.
- Retired identity remains read-only.

## Non-claims

- Does not simulate every organizational workflow.
- Does not use internal state mutation shortcuts.
