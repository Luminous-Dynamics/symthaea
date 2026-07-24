# Patch 0014: feat incident add reopening review package

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Produce a portable package for independent review of the challenge-to-freeze chain.

## Intended changes

- Bundle original incident, closure, segment, challenge ledger, adverse-evidence report, policy, authorization, freeze receipt, and limitations.
- Include exact verifier identities, expected policies, canonical bytes, and manifests.
- Support offline verification without requiring operational telemetry.

## Required tests

- Missing, extra, substituted, or divergent package objects fail.
- The package reproduces byte-for-byte.
- Review acceptance distinguishes technical trigger, governed authorization, and committed freeze.

## Non-claims

- Does not prove universal fork knowledge.
- Does not disclose private evidence unless the release policy permits it.
