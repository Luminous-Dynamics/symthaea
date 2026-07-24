# Patch 0018: Publish bounded-verification release evidence

**Series:** 24

## Objective

Document resource profiles, guarantees, limits, and non-claims for hostile public artifacts.

## Intended changes

- Publish limit-policy schema, malicious corpus identity, conformance results, worst-case-valid benchmarks, archive-safety report, and operator guidance.
- State which limits are defaults versus deployment choices.
- Package all artifacts deterministically and verify them with the Series 23 release lane.

## Required tests

- Public kit contains no active bomb payload requiring unsafe extraction.
- All checksums and corpus identities verify.
- Series 23 clean-room reproduction remains green after Series 24.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
