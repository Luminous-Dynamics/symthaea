# Graceful Arbitration Recovery Protocol

## Purpose

Resource scarcity is not sufficient evidence that an allocation process will eventually make progress. This protocol adds a liveness authority below protected safety constraints and above discretionary mission work.

## Authority order

1. Physical hazards and final-command invariants.
2. Protected return and environmental containment.
3. Resource-conflict admission.
4. Arbitration-liveness recovery.
5. Operator, team, and mission preferences.

The liveness layer never creates actuator authority. It may only reduce productive work, require protected return, or hold for review.

## Progress evidence

A frame counts as material progress only when at least one bounded operational fact changes meaningfully: a work order completes, restoration advances, hazard severity falls, return margin improves, or the selected objective set changes. Numeric noise does not reset the deadlock timer.

## Recovery sequence

- Nominal: retain ordinary resource-conflict authority.
- Warning: shed discretionary objectives deterministically and clamp productive demand.
- Critical with feasible return: stop productive work and preserve withdrawal.
- Critical without feasible return: stop movement and retain recovery actuators for accountable review.

Protected objectives are never shed.

## Persistence

Deadlock history and recovery attempts are persisted in operational checkpoint schema 13. Restart cannot erase unresolved no-progress debt.

## Non-claims

This bounded monitor does not prove global mission liveness, optimal allocation, stakeholder legitimacy, or physical reachability. Those claims require full-workspace simulation, HIL trials, calibrated progress signals, and independent operations review.
