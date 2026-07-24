# Patch 0005: feat add cycle two resumption policy context

**Series:** 34

## Objective

Implement verifier-owned resumption requirements specific to a later recovery cycle.

## Intended changes

- Represent minimum post-certification advance, accepted authority and witness roles, closure freshness, quarantine constraints, delegation and allowance freshness, and authorization expiry.
- Bind the expected cycle and successor segment.
- Prevent artifact-supplied weakening.

## Acceptance evidence

- Policy substitution, wrong cycle, and weaker embedded thresholds fail.
- Unknown fields fail closed.
- Policy identity is included in reports and caches.

## Non-claims

- Does not choose a universal resumption policy.
- Does not authenticate signers.
