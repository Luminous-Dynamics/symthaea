# Patch 0005: feat incident evaluate objective reopening triggers

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Produce reproducible adverse-evidence reports before any authority decision.

## Intended changes

- Verify referenced artifacts through their native Series 16–22 contracts.
- Report structural validity, authentication, contradiction, independence assumptions, policy satisfaction, and unresolved ambiguity separately.
- Bind the report to the expected trigger policy and exact current segment/head.

## Required tests

- Forged equivocation, wrong branch, stale head, policy substitution, and verifier-role confusion fail.
- Rust and independent verifier disagreement remains unresolved rather than majority-voted.
- A healthy operational snapshot cannot suppress an adverse evidence report.

## Non-claims

- Does not mutate closure, segment, or catalog state.
- Does not determine the final governance response.
