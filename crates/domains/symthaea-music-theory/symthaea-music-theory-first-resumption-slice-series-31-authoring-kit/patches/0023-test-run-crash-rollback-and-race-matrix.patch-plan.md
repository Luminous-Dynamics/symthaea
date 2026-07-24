# Patch 0023: test run crash rollback and race matrix

**Series:** 31

## Objective

Qualify the slice's transaction behavior under failure and concurrency.

## Intended changes

- Inject failure before and after each staged change.
- Race two valid first-mutation attempts.
- Restart after each interruption.

## Acceptance evidence

- Exactly zero or one commit is visible.
- Failed cases restore byte-identical pre-state.
- No allowance or ordinal is partially advanced.

## Non-claims

- Does not benchmark distributed systems.
- Does not model every filesystem failure.
