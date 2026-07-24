# Patch 0003: refactor extend shared identities for cycle two

**Series:** 33

## Objective

Add the minimal canonical identities required for a repeated recovery cycle.

## Intended changes

- Add recovery-cycle, cycle-ledger event, cycle authority epoch, cycle witness policy, quarantine transition, branch candidate, recovery plan, authorization, selection receipt, certification, closure, and lifecycle-report identities.
- Bind every identity to exact cycle and predecessor evidence.
- Preserve Series 31–32 canonical bytes.

## Acceptance evidence

- Cross-cycle and cross-role replay vectors fail.
- Fixed-width canonical vectors are frozen.
- Cycle ordinal and predecessor linkage are cross-checked.

## Non-claims

- Does not add retirement identities.
- Does not make cycle identity canonical branch authority.
