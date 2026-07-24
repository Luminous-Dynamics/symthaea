# Patch 0005: Bound catalog, event, status-proof, and lineage work

**Series:** 24

## Objective

Place explicit limits on the append-only structures whose valid size can grow over time.

## Intended changes

- Cap records, events, status proofs, segments, checkpoints, and lineage hops per verification invocation.
- Use iterative traversal and visited sets with bounded capacity.
- Separate too-large-for-profile from structurally invalid.

## Required tests

- Long cycles fail structurally without recursion overflow.
- A valid artifact exactly at each limit passes.
- One-over-limit artifacts fail with stable codes.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
