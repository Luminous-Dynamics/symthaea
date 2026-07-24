# Patch 0017: feat enforce global publication and event ordinals

**Series:** 31

## Objective

Continue the catalog's global numbering across the recovery boundary.

## Intended changes

- Read predecessor ordinals from the Series 21 baseline catalog.
- Require exact next values for publication and event append.
- Cross-check ordinals in receipt and audit.

## Acceptance evidence

- Reset, duplicate, gap, and regression fixtures fail.
- Successful commit advances each applicable ordinal exactly once.
- Segment-local counters cannot replace global order.

## Non-claims

- Does not make ordinals trusted time.
- Does not infer external catalog completeness.
