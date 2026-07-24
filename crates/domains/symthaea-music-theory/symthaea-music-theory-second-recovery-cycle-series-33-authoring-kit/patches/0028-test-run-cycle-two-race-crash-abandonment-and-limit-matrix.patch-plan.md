# Patch 0028: test run cycle two race crash abandonment and limit matrix

**Series:** 33

## Objective

Qualify branch-selection and closure transactions under failure and concurrency.

## Intended changes

- Race competing branch selections and closures, inject failure at every stage, restart after interruption, exercise active-attempt limits, and replay abandoned attempts.
- Verify byte-identical rollback.
- Retain deterministic seeds.

## Acceptance evidence

- Exactly zero or one conflicting transition commits.
- Abandoned attempts remain terminal.
- No partial cycle, quarantine, or checkpoint state is visible.

## Non-claims

- Does not model every distributed storage failure.
- Does not benchmark production throughput.
