# Patch 0002: test add qualified active lineage baseline

**Series:** 35

## Objective

Create the exact active two-cycle baseline consumed by retirement.

## Intended changes

- Package all incidents, cycles, closures, segments, first-mutation receipts, catalog head, global ordinals, active policies, quarantines, delegations, allowances, and audits.
- Use synthetic identities.
- Include incomplete-history, stale-head, and hidden-capability variants.

## Acceptance evidence

- The positive fixture passes all prior audits.
- Mutated variants fail at stable stages.
- The baseline archive reproduces byte-for-byte.

## Non-claims

- Does not claim prior slices are implemented in the canonical repository.
- Does not include production secrets.
