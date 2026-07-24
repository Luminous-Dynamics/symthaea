# Patch 0027: test run retirement policy replay and staleness corpus

**Series:** 35

## Objective

Freeze retirement authorization and policy attacks.

## Intended changes

- Cover weakened policy, missing history, stale trigger report, reused closure/recovery/reopening/resumption signatures, duplicate roles, quarantined signers, wrong head, and omitted active capability.
- Run native and independent verification.
- Require stable stages and issue codes.

## Acceptance evidence

- Trigger satisfaction alone never retires.
- No replayed signature counts.
- Valid threshold-edge authorization succeeds.

## Non-claims

- Does not prove every compromise is detectable.
- Does not replace external identity governance.
