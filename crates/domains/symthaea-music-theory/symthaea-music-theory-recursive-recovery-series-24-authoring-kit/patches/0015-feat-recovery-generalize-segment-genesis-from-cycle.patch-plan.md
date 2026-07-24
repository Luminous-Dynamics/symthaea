# Patch 0015: feat recovery generalize segment genesis from cycle

**Series:** 24

## Objective

Allow Series 22 trust-segment creation to consume any accepted cycle closure rather than only the first recovery.

## Intended changes

- Bind new segment genesis to the exact cycle identity, closure bundle, post-cycle certification, predecessor frozen segment, and global catalog ordinals.
- Require a new segment identity after every successful repeated recovery.
- Preserve prior segment membership and status events.

## Required tests

- Reusing an earlier segment ID or closure fails.
- Segment predecessor and cycle predecessor must agree.
- Global ordinals never reset.

## Non-claims

- Does not permit publication before a separate resumption authorization.
- Does not merge segments.
