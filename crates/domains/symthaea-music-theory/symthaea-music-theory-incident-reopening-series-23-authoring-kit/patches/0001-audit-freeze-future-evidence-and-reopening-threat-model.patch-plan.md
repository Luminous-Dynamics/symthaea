# Patch 0001: audit freeze future evidence and reopening threat model

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Turn Series 21's FutureEvidenceMayReopenTheIncident limitation into an explicit, conservative lifecycle.

## Intended changes

- Inventory evidence that may arrive after closure: hidden branch proof, witness equivocation, mirror conflict, corrupted preservation copy, verifier disagreement, policy compromise, and invalid resumed mutation.
- Separate untrusted challenge intake, technically corroborated adverse evidence, governed reopening authorization, and actual publication freeze.
- Model challenge spam, forged evidence, stale closure, wrong segment, repeated reopening, and incident-history erasure.

## Required tests

- Every proposed trigger has an evidence source, verification path, authority boundary, and expected operational consequence.
- Closure remains historically valid as a past decision even when later superseded operationally.
- No untrusted submission directly mutates publication state.

## Non-claims

- Does not assert every allegation warrants reopening.
- Does not assign legal or personal fault.
