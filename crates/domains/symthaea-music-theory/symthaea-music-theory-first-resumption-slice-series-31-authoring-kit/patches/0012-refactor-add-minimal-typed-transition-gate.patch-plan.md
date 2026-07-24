# Patch 0012: refactor add minimal typed transition gate

**Series:** 31

## Objective

Centralize commit-time checks for the first slice.

## Intended changes

- Check closure remains accepted, segment is active, plan and authorization match, policies are current, delegation and allowance are valid, quarantine state is acceptable, and pre-head matches.
- Return stable stage and issue codes.
- Avoid consulting telemetry or documentation.

## Acceptance evidence

- Every stale-state mutation fails before commit.
- Earliest-failure ordering is deterministic.
- No mutation path bypasses the gate in the slice.

## Non-claims

- Does not yet cover reopening or retirement.
- Does not persist state.
