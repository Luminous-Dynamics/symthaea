# Patch 0008: feat implement cycle two dual quorum resumption

**Series:** 34

## Objective

Verify the successor-segment plan under active cycle-two recovery-authority and witness policies.

## Intended changes

- Count both quorums independently.
- Exclude stale, duplicate, quarantined, wrong-role, wrong-cycle, and externally rejected statements.
- Report each threshold and exclusion separately.

## Acceptance evidence

- Threshold-edge valid authorization succeeds.
- Earlier-cycle statements never count.
- Authorization cannot be reused for another segment or publication.

## Non-claims

- Does not mutate state.
- Does not prove signer independence.
