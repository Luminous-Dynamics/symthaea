# Patch 0008: feat implement multi role retirement authorization

**Series:** 35

## Objective

Verify the exact retirement plan under verifier-owned expected thresholds.

## Intended changes

- Count configured roles separately and enforce independence requirements.
- Exclude duplicate, stale, quarantined, wrong-role, and externally rejected statements.
- Report normal and emergency policy variants explicitly.

## Acceptance evidence

- Threshold-edge valid authorization succeeds.
- Weaker embedded policy never counts.
- Authorization cannot be reused for another head or lineage.

## Non-claims

- Does not commit retirement.
- Does not prove organizational independence.
