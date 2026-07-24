# Patch 0002: feat retirement model caller owned trust exhaustion policy

**Series:** 25

## Objective

Let deployments define conservative terminal thresholds without allowing artifacts to weaken them.

## Intended changes

- Add fixed-width policy dimensions for maximum completed recovery cycles, maximum reopenings, unresolved verifier disagreements, compromised authority classes, forbidden quarantine states, and mandatory human review.
- Support immediate-retirement conditions for explicitly configured catastrophic failures.
- Bind the expected policy identity into all trigger and authorization reports.

## Required tests

- Artifact-supplied values cannot relax expected policy.
- Unknown dimensions fail closed.
- Boundary and combination rules are deterministic.

## Non-claims

- Does not use a universal risk score.
- Does not automatically retire solely because a counter increased.
