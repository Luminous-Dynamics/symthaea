# Patch 0008: feat incident model segment freeze and reopen receipts

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Record the exact point at which the operational segment stops accepting publication mutations.

## Intended changes

- Add planned and committed freeze receipts binding the reopening authorization, active segment, pre-freeze head, segment state event, quarantine actions, and resulting operating status.
- Keep the prior closure and first-mutation receipts referenced immutably.
- Require explicit reason classes without free-form blame fields in canonical authority bytes.

## Required tests

- Wrong head, wrong segment, changed authorization, or changed quarantine action breaks the receipt.
- A plan receipt cannot masquerade as committed freeze evidence.
- Committed freeze is idempotent and cannot be applied to another segment.

## Non-claims

- Does not delete or invalidate historical publications automatically.
- Does not prove the incident root cause.
