# Patch 0016: feat implement archive only operating mode

**Series:** 35

## Objective

Expose immutable lifecycle evidence after mutation authority ends.

## Intended changes

- Provide read-only access to catalog, incidents, cycles, segments, closures, receipts, policies, manifests, vectors, and limitations.
- Reject implicit repair, normalization, or synthetic reconstruction.
- Report completeness and missing objects explicitly.

## Acceptance evidence

- Archive verification works without active signing capability.
- Missing objects render incomplete rather than healthy.
- Read-only APIs cannot mutate state.

## Non-claims

- Does not guarantee permanent hosting.
- Does not make the archive globally canonical.
