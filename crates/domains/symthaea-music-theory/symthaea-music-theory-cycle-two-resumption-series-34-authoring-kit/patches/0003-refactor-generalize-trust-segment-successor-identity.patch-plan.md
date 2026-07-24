# Patch 0003: refactor generalize trust segment successor identity

**Series:** 34

## Objective

Allow a new segment to derive from any accepted recovery-cycle closure while preserving predecessor segments.

## Intended changes

- Bind cycle identity, closure, certification, selected branch, predecessor frozen segment, catalog head, policies, and predecessor segment.
- Retain the Series 31 genesis encoding for first-cycle compatibility.
- Use explicit versioned successor semantics.

## Acceptance evidence

- Reusing the prior segment ID fails.
- Wrong cycle, closure, certification, predecessor, or head changes identity or fails.
- Series 31 vectors remain unchanged.

## Non-claims

- Does not establish global branch canonicality.
- Does not activate publication.
