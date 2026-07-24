# Patch 0013: refactor extend transition gate for cycle selection

**Series:** 33

## Objective

Add frozen-state, cycle, candidate, policy, and quarantine preconditions to the shared transition gate.

## Intended changes

- Check exact frozen segment and head, cycle-ledger state, active attempt slot, plan and authorization, candidate set, authority and witness policies, and quarantines.
- Detect publication, retirement, or competing-recovery state changes.
- Use deterministic earliest failure.

## Acceptance evidence

- Stale head, changed freeze, candidate mutation, policy rotation, new quarantine, and competing attempt fail before commit.
- No telemetry or operator note influences authority.
- All cycle selection mutations use the gate.

## Non-claims

- Does not cover final retirement.
- Does not persist state.
