# Patch 0010: refactor add single typed transition gate

**Series:** 26

## Objective

Prevent mutation paths from drifting into inconsistent precondition logic.

## Intended changes

- Add a typed transition-gate layer that evaluates catalog head, segment, incident, cycle, retirement, quarantine, policy, delegation, allowance, and expected-operation state.
- Return stable ordered failure stages and evidence references.
- Keep transition-specific authorization in role-specific code.

## Required tests

- Every authoritative mutation path calls the gate at commit time.
- Inventory tests fail when a new mutation bypasses it.
- Earliest-failure behavior is deterministic.

## Non-claims

- Does not turn the gate into a universal authorization token.
- Does not accept telemetry as a precondition.
