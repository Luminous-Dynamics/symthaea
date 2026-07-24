# Patch 0011: refactor extend transition gate for successor resumption

**Series:** 34

## Objective

Add cycle-two closure, successor segment, and fresh-capability checks to the typed gate.

## Intended changes

- Reverify cycle-two closure and certification, active successor segment, exact head, policies, quarantines, authorization, delegation, allowance, and global ordinals.
- Detect reopening, retirement, or catalog mutation races.
- Retain deterministic earliest-failure ordering.

## Acceptance evidence

- Changed cycle state, reopened incident, new quarantine, policy rotation, stale head, and consumed allowance fail before commit.
- No telemetry or review state influences authority.
- All successor first-mutation paths use the gate.

## Non-claims

- Does not persist state.
- Does not cover ordinary later publications.
