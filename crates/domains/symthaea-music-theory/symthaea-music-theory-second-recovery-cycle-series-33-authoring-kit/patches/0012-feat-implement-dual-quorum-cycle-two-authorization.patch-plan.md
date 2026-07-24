# Patch 0012: feat implement dual quorum cycle two authorization

**Series:** 33

## Objective

Verify exact branch-selection authorization under active cycle-two policies.

## Intended changes

- Count recovery-authority and witness quorums separately.
- Exclude stale, duplicate, quarantined, wrong-role, wrong-cycle, and externally rejected signers.
- Report every exclusion and threshold independently.

## Acceptance evidence

- Threshold-edge valid sets succeed.
- Cycle-one authorization cannot satisfy cycle two.
- Authorization cannot be reused for another candidate.

## Non-claims

- Does not commit selection.
- Does not prove branch canonicality.
