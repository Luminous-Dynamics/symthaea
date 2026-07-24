# Patch 0010: refactor extend transition gate for retirement

**Series:** 35

## Objective

Add terminal-retirement preconditions and a hard retired-lineage check to the shared gate.

## Intended changes

- Reverify trigger report, plan, authorization, exact head, active segment, cycles, quarantines, capabilities, policies, and pending transitions.
- Block all later mutation gates once retirement is committed.
- Use deterministic earliest-failure ordering.

## Acceptance evidence

- Stale state, changed capability inventory, new transition, policy rotation, or missing revocation fails before commit.
- Cached healthy state cannot bypass retirement.
- All mutation paths consult retired-lineage state.

## Non-claims

- Does not persist state.
- Does not control old external binaries.
