# Patch 0012: feat recovery model post cycle fresh checkpoint

**Series:** 24

## Objective

Require the selected branch to advance beyond the later-cycle recovery anchor before re-entry.

## Intended changes

- Add cycle-bound certification input for exact branch continuity, catalog advance, active recovered witness policy, authority activation, and quarantine state.
- Require a strictly later checkpoint and configured minimum advance.
- Preserve the complete predecessor cycle and segment lineage.

## Required tests

- Same-head, pre-anchor, wrong-cycle, wrong-branch, insufficient-advance, and stale-policy cases fail.
- Freshness does not rely on local mtimes.
- Certification can be independently reconstructed.

## Non-claims

- Does not authorize publication.
- Does not prove all forks are known.
