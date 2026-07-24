# Patch 0019: feat add quarterly maintenance scorecard

**Series:** 29

## Objective

Make maintenance health visible without reducing correctness to one percentage.

## Intended changes

- Report open blockers, reproduced regressions, fixture promotion, conformance status, reproduction status, advisories, deprecations, soak results, game days, and overdue reviews.
- Distinguish unknown, not run, failed, waived, and passed.
- Link every status to evidence.

## Required evidence

- Missing collectors render unknown rather than healthy.
- Aggregate scores cannot hide a blocker.
- Scorecard generation is deterministic.

## Non-claims

- Does not make metrics evidence authority.
- Does not guarantee future maintenance quality.
