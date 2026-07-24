# Patch 0015: feat implement shell free external verifier roles

**Series:** 26

## Objective

Make all new signatures and technical evidence verifiable through bounded typed subprocess contracts.

## Intended changes

- Add request and response roles for resumption, reopening, cycle recovery, retirement, branch contradiction, equivocation, and terminal observers.
- Pass exact canonical bytes and expected identities through standard input/output without shell evaluation.
- Retain timeout, output-size, process-count, and cancellation limits.

## Required tests

- Wrong role, target, signer, response, timeout, oversized output, and nonzero exit fail safely.
- External verifier execution cannot mutate state.
- Shell metacharacters are treated as data.

## Non-claims

- Does not manage private keys.
- Does not choose cryptographic algorithms automatically.
