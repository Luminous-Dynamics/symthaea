# Patch 0020: test incident cover hidden fork equivocation and invalid resumption

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Freeze the highest-consequence objective reopening triggers.

## Intended changes

- Cover authenticated witness equivocation, exact continuity contradiction, hidden-branch evidence, invalid preservation lineage, and invalid first-resumed mutation receipt.
- Exercise single and corroborated trigger policies.
- Require Rust and independent verifier agreement.

## Required tests

- Valid adverse evidence satisfies only the configured technical trigger.
- Forged or wrong-lineage evidence does not count.
- Trigger satisfaction alone does not freeze publication.

## Non-claims

- Does not claim all hidden forks are detectable.
- Does not resolve ambiguous independent-verifier disagreement automatically.
