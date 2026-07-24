# Patch 0017: Report verification resource usage

**Series:** 24

## Objective

Give operators and offline verifiers enough evidence to tune limits and diagnose rejection.

## Intended changes

- Emit bounded structured reports with configured limits, observed maxima, external-call counts, archive expansion, stage timings, and cancellation state.
- Exclude secrets, private governance data, raw credentials, and unbounded child output.
- Bind reports to exact artifact and verifier identities.

## Required tests

- Report numbers match instrumented counters.
- Reports themselves obey size limits.
- A report cannot be mistaken for an authorization decision.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
