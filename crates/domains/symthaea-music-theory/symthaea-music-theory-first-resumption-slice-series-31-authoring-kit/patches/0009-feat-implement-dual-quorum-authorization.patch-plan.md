# Patch 0009: feat implement dual quorum authorization

**Series:** 31

## Objective

Verify one exact authorization set under caller-owned expected policy.

## Intended changes

- Count recovery-authority and witness quorums separately.
- Exclude duplicate, stale, quarantined, wrong-role, and externally rejected statements.
- Report structural, authentication, and policy results separately.

## Acceptance evidence

- Threshold-edge valid sets succeed.
- Bundle-supplied weaker thresholds do not count.
- Authorization cannot be reused for another plan.

## Non-claims

- Does not mutate catalog state.
- Does not make witnesses publishers.
