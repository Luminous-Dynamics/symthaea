# Patch 0017: feat tooling extend external verifier for reopening

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Support shell-free verification of challenge evidence and reopening signatures.

## Intended changes

- Add typed request roles for equivocation, branch contradiction, policy compromise, resumed-mutation validation, and authorization signatures.
- Pass exact expected policy and target identities.
- Retain bounded process execution and output parsing.

## Required tests

- Wrong role, wrong target, malformed response, timeout, excessive output, and nonzero exit fail safely.
- External verifier execution cannot append challenge or freeze state.
- Shell metacharacters are never interpreted.

## Non-claims

- Does not manage verifier enrollment.
- Does not trust external verifier self-description without local configuration.
