# Patch 0003: feat retirement evaluate trust exhaustion triggers

**Series:** 25

## Objective

Produce a reproducible technical report showing whether configured retirement conditions are met.

## Intended changes

- Evaluate the complete incident, recovery-cycle, segment, quarantine, authority, witness, verifier, and preservation history.
- Report satisfied, unsatisfied, unknown, waived-by-authority, and unsupported conditions separately.
- Bind the report to one exact catalog head and expected policy.

## Required tests

- Missing history renders unknown rather than safe.
- Healthy SLOs or alerts cannot suppress a satisfied condition.
- Policy changes invalidate cached trigger reports.

## Non-claims

- Does not mutate publication state.
- Does not make trigger satisfaction a retirement decision.
