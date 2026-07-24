# Patch 0022: test incident cover challenge spam privacy and resource limits

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Freeze malicious intake and disclosure cases.

## Intended changes

- Cover oversized context, excessive references, archive bombs, path traversal, verifier fan-out, duplicate challenges, secret fields, and retention-policy boundaries.
- Require stable resource and privacy failure codes.
- Verify public review packages omit prohibited fields.

## Required tests

- No limit failure partially appends state.
- Metrics and alerts remain low cardinality even under repeated challenges.
- Valid boundary-size challenges remain processable.

## Non-claims

- Does not claim the corpus covers all denial-of-service techniques.
- Does not define legal disclosure obligations.
