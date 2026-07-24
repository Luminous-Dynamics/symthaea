# Patch 0005: feat implement trust segment genesis

**Series:** 31

## Objective

Create a content-derived post-closure trust segment for the slice.

## Intended changes

- Bind Series 21 closure, post-recovery certification, selected branch, catalog head, policy context, and predecessor lineage.
- Add structural validation and canonical identity.
- Expose a minimal curated constructor.

## Acceptance evidence

- Wrong closure, branch, head, or policy changes the identity or fails.
- Self-predecessor and disconnected lineage are rejected.
- Vectors are architecture independent.

## Non-claims

- Does not activate publication.
- Does not establish universal canonicality.
