# Patch 0007: feat incident add dual quorum reopening authorization

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Require active recovery authorities and recovered witnesses to authorize reopening under the expected policy.

## Intended changes

- Verify exact plan signatures against the active policy epochs at the freeze head.
- Exclude quarantined, stale, duplicate, wrong-role, and externally rejected signers.
- Allow emergency policy variants only when independently configured and explicitly surfaced.

## Required tests

- Closure signatures, resumption signatures, and old recovery signatures cannot be replayed.
- Each quorum and any emergency exception are reported independently.
- Emergency authorization cannot silently downgrade the ordinary policy.

## Non-claims

- Does not prove signer independence.
- Does not make the challenge submitter a reopening authority.
