# Patch 0011: Add transactional and cancellation-safe workflow

**Series:** 24

## Objective

Ensure interrupted verification cannot leave accepted partial state or reusable temporary artifacts.

## Intended changes

- Stage results in temporary state and commit only after all required dimensions pass.
- Propagate cancellation through hashing, lineage, external calls, and archive handling.
- Use RAII cleanup for temporary directories and files.

## Required tests

- Cancellation at every stage leaves no accepted record.
- Crash-recovery test finds no partial commit.
- Temporary artifacts are removed or clearly quarantined.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
