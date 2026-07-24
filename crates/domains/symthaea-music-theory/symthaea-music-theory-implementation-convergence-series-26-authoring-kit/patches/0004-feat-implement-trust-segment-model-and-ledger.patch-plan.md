# Patch 0004: feat implement trust segment model and ledger

**Series:** 26

## Objective

Convert the Series 22 trust-segment plans into compiled, audited domain code.

## Intended changes

- Implement content-derived segment identity, genesis, predecessor linkage, ledger events, active lookup, and full audit.
- Bind first post-recovery genesis to exact Series 21 closure and certification inputs.
- Preserve global catalog ordinal continuity outside segment-local state.

## Required tests

- Genesis, append, mutation, reordering, and disconnected-lineage vectors pass.
- At most one segment is active at one exact head.
- Historical records remain verifiable without implicit reassignment.

## Non-claims

- Does not authorize publication.
- Does not establish universal branch canonicality.
