# Patch 0004: feat add verifier owned resumption policy context

**Series:** 31

## Objective

Implement the expected-policy object supplied by the caller or deployment.

## Intended changes

- Represent thresholds, accepted signer roles, freshness, minimum advance, expiry, and delegation/allowance requirements.
- Prevent bundle-supplied policy from weakening it.
- Bind policy identity into every report.

## Acceptance evidence

- Policy substitution and downgrade fixtures fail.
- Unknown fields fail closed.
- Cache keys include the exact expected policy.

## Non-claims

- Does not choose one universal policy.
- Does not authenticate signers by itself.
