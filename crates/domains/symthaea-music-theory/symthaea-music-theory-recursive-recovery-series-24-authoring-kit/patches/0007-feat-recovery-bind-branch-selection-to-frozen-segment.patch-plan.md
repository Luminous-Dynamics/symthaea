# Patch 0007: feat recovery bind branch selection to frozen segment

**Series:** 24

## Objective

Anchor later recovery to the exact segment and catalog head frozen by Series 23.

## Intended changes

- Require the reopening freeze receipt, frozen segment identity, exact freeze head, challenge ledger, and adverse-evidence report in every recovery plan.
- Reject branch candidates that predate the freeze anchor or omit the incident recurrence lineage.
- Preserve non-selected candidates and rejection reasons append-only.

## Required tests

- Wrong segment, wrong freeze receipt, pre-freeze candidate, and hidden predecessor substitutions fail.
- Selecting one branch does not delete other candidate evidence.
- Candidate ordering cannot change the selected identity.

## Non-claims

- Does not establish universal branch canonicality.
- Does not infer branch correctness from length alone.
