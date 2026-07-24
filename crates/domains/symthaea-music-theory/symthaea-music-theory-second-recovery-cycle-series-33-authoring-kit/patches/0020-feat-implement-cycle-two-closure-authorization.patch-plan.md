# Patch 0020: feat implement cycle two closure authorization

**Series:** 33

## Objective

Verify a distinct dual-quorum decision to close the second recovery cycle.

## Intended changes

- Add closure-specific recovery-authority and witness statements.
- Require active cycle-two policies and exact closure plan.
- Exclude stale, duplicate, quarantined, wrong-role, and externally rejected signers.

## Acceptance evidence

- Cycle-one closure and cycle-two recovery signatures cannot replay.
- Threshold-edge valid authorization succeeds.
- Authorization is bound to one exact checkpoint and cycle.

## Non-claims

- Does not commit closure.
- Does not authorize a new segment.
