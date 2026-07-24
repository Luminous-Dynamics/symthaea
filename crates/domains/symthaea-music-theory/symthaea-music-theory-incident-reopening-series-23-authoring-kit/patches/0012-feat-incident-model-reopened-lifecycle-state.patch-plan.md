# Patch 0012: feat incident model reopened lifecycle state

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Expose the complete incident state without reducing it to a boolean open/closed flag.

## Intended changes

- Represent investigating, contained, recovered, re-entered, closed, resumed, challenged, reopening-authorized, frozen, and superseded-by-later-recovery states.
- Derive current state from append-only evidence rather than a mutable status field.
- Report inconsistent or unsupported histories explicitly.

## Required tests

- Contradictory closure/freeze/resumption histories render inconsistent.
- Later recovery does not erase the reopened period.
- State derivation is deterministic under event ordering.

## Non-claims

- Does not claim one lifecycle fits every external governance process.
- Does not replace underlying evidence reports.
