# Patch 0009: refactor unify verifier owned expected policy context

**Series:** 26

## Objective

Remove bundle-controlled policy acceptance across all new verification paths.

## Intended changes

- Introduce explicit expected-policy context for witness, recovery, resumption, reopening, cycle, retirement, preservation, and external-verifier checks.
- Bind policy identity and verification epoch into cache keys and reports.
- Report embedded-policy satisfaction separately from trusted-policy satisfaction.

## Required tests

- Bundle-supplied threshold downgrade is rejected everywhere.
- Policy changes invalidate cached results.
- Missing expected policy cannot produce acceptance.

## Non-claims

- Does not prescribe one global policy.
- Does not make policy identity secret.
