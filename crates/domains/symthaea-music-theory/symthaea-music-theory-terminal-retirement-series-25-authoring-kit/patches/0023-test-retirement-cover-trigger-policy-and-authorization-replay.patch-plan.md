# Patch 0023: test retirement cover trigger policy and authorization replay

**Series:** 25

## Objective

Freeze retirement-policy substitution and signature-replay attacks.

## Intended changes

- Cover artifact-weakened policy, stale trigger report, reused closure/recovery/reopening signatures, duplicate roles, quarantined signer, wrong head, and omitted active attempt.
- Require stable issue codes.
- Exercise threshold-edge valid retirement.

## Required tests

- No replayed signature counts.
- Trigger satisfaction alone cannot retire.
- Policy changes invalidate cached reports.

## Non-claims

- Does not prove every compromise is detectable.
- Does not replace external identity governance.
