# Patch 0013: security incident bound challenge storage and verification

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Prevent reopening intake from becoming a resource-exhaustion or privacy attack.

## Intended changes

- Apply caller-owned byte, object, depth, reference, verifier-call, and retention limits.
- Require streaming digest verification for large referenced artifacts.
- Redact or segregate submitter context from public evidence packages.

## Required tests

- Compression bombs, path traversal, duplicate-object amplification, deep nesting, and verifier fan-out are rejected under stable resource codes.
- Limit failures do not partially append challenge state.
- Public exports contain no prohibited submitter or credential material.

## Non-claims

- Does not guarantee anonymity against all auxiliary information.
- Does not select legal retention periods.
