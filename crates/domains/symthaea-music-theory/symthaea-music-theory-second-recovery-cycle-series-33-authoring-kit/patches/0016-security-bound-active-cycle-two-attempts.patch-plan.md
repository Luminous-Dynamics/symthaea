# Patch 0016: security bound active cycle two attempts

**Series:** 33

## Objective

Prevent branch-candidate and recovery-plan multiplication from exhausting resources or amplifying authority.

## Intended changes

- Apply caller-owned bounds to candidate count, plans, signatures, external verifier calls, and concurrent attempts.
- Require explicit abandonment receipts.
- Reject duplicate candidate amplification.

## Acceptance evidence

- Limit failures create no partial state.
- Abandoned attempts cannot later commit.
- Boundary-sized valid attempts remain processable.

## Non-claims

- Does not prescribe a universal attempt count.
- Does not decide terminal retirement.
