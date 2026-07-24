# Patch 0017: feat implement post cycle two fresh checkpoint input

**Series:** 33

## Objective

Represent the strictly later checkpoint required after branch selection.

## Intended changes

- Bind cycle identity, selected branch, recovery anchor, later catalog head, continuity proof, witness policy, mirror/gossip evidence where present, and quarantines.
- Require configured minimum catalog advance.
- Expose structural validation independently from authentication.

## Acceptance evidence

- Same-head, pre-anchor, wrong-cycle, wrong-branch, insufficient-advance, and stale-policy variants fail.
- No local mtime is used.
- Input reconstructs independently.

## Non-claims

- Does not authenticate the checkpoint.
- Does not authorize closure.
