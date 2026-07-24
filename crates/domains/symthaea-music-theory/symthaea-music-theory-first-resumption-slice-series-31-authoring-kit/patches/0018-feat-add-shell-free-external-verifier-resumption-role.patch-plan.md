# Patch 0018: feat add shell free external verifier resumption role

**Series:** 31

## Objective

Support bounded external authentication for resumption statements and delegation.

## Intended changes

- Define typed request and response roles over exact canonical payload bytes.
- Pin executable and expected policy identity.
- Apply timeout, output-size, and exit-status bounds.

## Acceptance evidence

- Shell metacharacters are not interpreted.
- Wrong role, malformed output, timeout, and policy mismatch fail safely.
- External verification cannot mutate state.

## Non-claims

- Does not choose cryptographic algorithms.
- Does not manage private keys.
