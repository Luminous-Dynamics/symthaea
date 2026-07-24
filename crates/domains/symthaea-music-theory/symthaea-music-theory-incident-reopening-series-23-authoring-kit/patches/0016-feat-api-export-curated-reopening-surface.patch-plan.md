# Patch 0016: feat api export curated reopening surface

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Expose challenge, trigger, authorization, freeze, recurrence, and lifecycle workflows through curated APIs.

## Intended changes

- Export supported models, builders, audits, verifier traits, transaction outcomes, and issue codes.
- Keep internal canonicalization helpers private.
- Add compile-oriented end-to-end examples.

## Required tests

- Existing public exports remain available.
- Examples compile using only curated paths.
- No private type leaks into a public signature.

## Non-claims

- Does not freeze internal module layout.
- Does not imply every challenge type is supported.
