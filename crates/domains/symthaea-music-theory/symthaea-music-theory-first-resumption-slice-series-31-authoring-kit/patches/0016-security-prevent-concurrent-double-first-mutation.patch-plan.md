# Patch 0016: security prevent concurrent double first mutation

**Series:** 31

## Objective

Ensure only one publication becomes the first mutation of the segment.

## Intended changes

- Use expected-head and first-mutation-slot compare-and-commit preconditions.
- Define retry and idempotency behavior.
- Return a stable conflict result.

## Acceptance evidence

- Two racing valid plans produce exactly one committed receipt.
- The loser does not consume allowance.
- A retry against the new head cannot become first.

## Non-claims

- Does not solve multi-datacenter consensus.
- Does not guarantee fairness among contenders.
