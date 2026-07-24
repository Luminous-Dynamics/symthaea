# Patch 0006: feat recovery model cycle scoped witness policy

**Series:** 24

## Objective

Require recovered witness policy and fresh observations to be attributable to the exact recovery cycle.

## Intended changes

- Bind witness policy epochs, activation points, checkpoint statements, and quarantines to cycle identity.
- Require a fresh checkpoint strictly after the cycle's branch-selection or recovery anchor.
- Preserve witness history from prior cycles without allowing it to satisfy new-cycle freshness.

## Required tests

- Old-cycle witness sets, pre-anchor observations, and wrong-policy statements do not count.
- Freshness and threshold acceptance are reported separately.
- Artifact-supplied policy cannot weaken verifier-supplied expected policy.

## Non-claims

- Does not prove witness independence.
- Does not turn freshness into trusted wall-clock time.
