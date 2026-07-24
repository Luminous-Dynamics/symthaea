# Patch 0003: refactor create shared lifecycle domain primitives

**Series:** 26

## Objective

Implement the minimal shared identities and references needed by resumption, reopening, recursive recovery, and retirement.

## Intended changes

- Add typed catalog-head, policy-epoch, segment, incident, cycle, closure, freeze, and terminal-lineage references.
- Centralize fixed-width ordinals, checked conversions, domain separators, and canonical identity helpers.
- Keep role-specific signed payloads distinct even when they share references.

## Required tests

- Cross-role payload substitution fails.
- All persisted counts and ordinals are fixed width.
- Existing Series 21 vectors remain unchanged.

## Non-claims

- Does not collapse distinct authorities into one generic signature type.
- Does not introduce state transitions yet.
