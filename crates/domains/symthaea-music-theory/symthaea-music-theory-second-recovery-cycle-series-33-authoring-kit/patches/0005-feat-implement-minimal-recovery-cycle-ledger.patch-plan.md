# Patch 0005: feat implement minimal recovery cycle ledger

**Series:** 33

## Objective

Persist cycle-two activation, progress, closure, abandonment, and terminal evidence append-only.

## Intended changes

- Add predecessor cycle reference, active-attempt slot, selected-branch receipt, certification reference, closure reference, and audit.
- Keep cycle-one history immutable.
- Derive current cycle state from events.

## Acceptance evidence

- Removal, reordering, duplicate active cycles, skipped ordinals, and state regression fail.
- At most one cycle-two attempt is active at the fixture head.
- Ledger bytes reproduce deterministically.

## Non-claims

- Does not merge separate incidents.
- Does not model unlimited simultaneous attempts.
