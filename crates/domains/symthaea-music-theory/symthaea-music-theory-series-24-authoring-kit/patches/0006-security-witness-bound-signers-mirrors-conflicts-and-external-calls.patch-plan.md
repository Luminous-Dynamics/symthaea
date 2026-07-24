# Patch 0006: Bound signer, mirror, conflict, and external authentication work

**Series:** 24

## Objective

Prevent signature-amplification and duplicate-identity attacks.

## Intended changes

- Reject duplicate signer/key/mirror identities before any external calls.
- Cap statements, policies, rotations, mirror observations, conflict proofs, and verifier invocations.
- Short-circuit only when doing so preserves the documented failure semantics.

## Required tests

- One million duplicate signers causes zero external calls.
- Call counts never exceed the configured budget.
- Threshold-edge valid sets still authenticate correctly.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
