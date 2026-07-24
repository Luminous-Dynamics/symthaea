# Patch 0027: test run cross cycle replay lineage and quarantine corpus

**Series:** 33

## Objective

Freeze cycle-confusion attacks for Series 33.

## Intended changes

- Cover cycle-one signatures, witness sets, certifications, closure statements, wrong predecessor, skipped ordinal, wrong frozen segment, wrong freeze receipt, cross-incident candidate, omitted quarantine, and unauthorized release.
- Run native and independent verification.
- Require stable stage and issue codes.

## Acceptance evidence

- Nothing from cycle one authorizes cycle two merely because it was valid.
- No quarantine disappears implicitly.
- Valid threshold-edge cases succeed.

## Non-claims

- Does not claim all signer compromise is detectable.
- Does not replace fuzzing.
