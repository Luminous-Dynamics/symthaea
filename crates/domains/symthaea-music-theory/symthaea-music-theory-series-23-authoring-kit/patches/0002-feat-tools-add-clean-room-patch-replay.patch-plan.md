# Patch 0002: Add clean-room patch replay tooling

**Series:** 23

## Objective

Reconstruct the cumulative tree from the exact pinned baseline without inheriting the author workspace.

## Intended changes

- Use a new temporary repository with sanitized Git configuration and environment.
- Apply numbered mail patches in declared order with no interactive fallback.
- Record commit IDs, tree IDs, patch IDs, stdout/stderr digests, and exit status.

## Required tests

- Replay succeeds twice in independent directories.
- A modified patch fails at the exact patch index.
- No global Git hooks or user configuration can affect the result.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
