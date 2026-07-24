# Formal Runtime Assurance Protocol

**System:** `symthaea-subterranean`  
**Campaign:** XVI  
**Date:** 2026-07-20  
**Status:** Executable internal assurance contract; not an external safety certification

## 1. Purpose

This protocol defines a bounded, executable assurance layer for the final
subterranean command path. It exists to detect contradictions between retained
runtime authority state and the command about to reach the plant, remove unsafe
authority in the same control frame, preserve a deterministic replay record,
and make the checked abstraction available to release evidence.

The protocol does **not** claim to prove the nonlinear plant, the learned
controller, the operating system, transport authentication, hardware timing,
or arbitrary memory safety. It checks a deliberately small abstraction that is
independent of the learned policy and safety-monotonic in operation.

## 2. Authority ordering

The formal assurance stage is placed after nominal policy selection, mission,
operator, stewardship, survivability, resource, maintenance, and actuator
constraints, but before the existing final-command invariant monitor and plant
actuation.

The relevant suffix of the command path is:

1. learned or reflex nominal command;
2. mission, hazard, operator, lifecycle, stewardship, and survivability limits;
3. physical actuator isolation;
4. **formal transition assurance**;
5. final-command invariant monitor;
6. plant actuation;
7. replay evidence append.

The formal stage may remove movement or productive-work authority. It may not
increase any actuator command. Thermal, dewatering, sealing, relay, and roof
support actions needed for safe containment remain available while a formal
hold is active.

## 3. Abstract runtime state

`FormalRuntimeState` is a bounded projection of the live platform containing:

- monotonically ordered control step;
- abstract safety level;
- whether a physical hazard is active;
- return-path feasibility;
- capability disposition;
- operator constraint;
- policy authority;
- terminal asset-decommission state;
- communications-partition authority;
- safe-service-location status;
- final candidate command.

The abstraction is intentionally smaller than the full embodiment. A property
not represented in this state is not proved by this protocol.

## 4. Executable transition contract

`FormalTransitionContract` evaluates the current abstract state, and where
available its predecessor, for contradictions including:

- malformed or unsupported state schema;
- non-finite or unbounded command values;
- step regression or a non-contiguous step gap;
- active hazard without safety escalation;
- productive work without a feasible return path;
- movement under `HoldForRecovery`;
- candidate policy authority without authorization;
- reversal of terminal decommissioning;
- motion while communications-partition authority is unavailable;
- work or motion inconsistent with operator constraints.

A detected violation is structured evidence. It is not reduced to an
unattributed Boolean.

## 5. Runtime monitor and recovery hold

`FormalAssuranceMonitor` is safety-monotonic:

- any transition violation latches a formal hold in the same frame;
- the hold zeros cutter, auger, tracks, and ballast motion;
- containment and recovery actuators remain available;
- the embodiment safety tier escalates to Red;
- the fallback is recorded as `FormalAssuranceHold`;
- the decision trace records a `FormalAssurance` authority stage.

The hold cannot clear from a single clean observation. Clearance requires a
bounded clean dwell at a safe service location. A new violation resets that
dwell immediately.

## 6. Deterministic replay ledger

`FormalReplayLedger` retains a bounded sequence of replay records containing:

- sequence and previous digest;
- abstract formal state;
- final command;
- decision-trace completeness;
- final invariant violation codes;
- formal transition violations.

Verification checks sequence continuity, digest linkage, chain head, frame
validity, state/record step agreement, and replay integrity after bounded
retention drops older records.

The included deterministic digest uses explicit wrapping arithmetic and exists
for reproducibility and corruption testing only. It is **not cryptographic**.
Production evidence must inject an approved cryptographic digest and secure
storage implementation.

## 7. Bounded model checking

`BoundedRuntimeModelChecker` exhaustively explores the finite abstraction for
six properties:

1. hazards require safety escalation;
2. infeasible return blocks productive work;
3. hold disposition blocks motion;
4. candidate control requires authority;
5. decommissioning is terminal;
6. control steps remain contiguous.

The current checker explores 32 bounded cases. Passing means that the
executable transition contract detects every preregistered counterexample in
that finite state space. It does not constitute a proof outside that space.

## 8. Adversarial evidence validation

`FormalAdversarialValidator` applies eight deterministic mutations:

- sequence replay;
- previous-digest replacement;
- final-command modification;
- state-step modification;
- trace-completeness modification;
- chain-head replacement;
- unauthorized candidate authority;
- decommission-state reversal.

Every mutation must be detected by either replay verification or transition
validation. The corpus is bounded and does not claim resistance to arbitrary
memory corruption, malicious compiler behavior, or cryptographic forgery.

## 9. Persistence and restart

Operational checkpoint schema version 9 includes the complete formal assurance
supervisor. Restoration validates monitor and replay state before activation.
Older checkpoint schemas default to no formal candidate authority and an empty,
valid replay history; they do not inherit a permissive hold-clear state.

A checkpoint cannot reverse terminal decommissioning, restore unauthorized
candidate authority, or silently accept an invalid replay chain.

## 10. Evidence and release gates

Five release-applicable requirements are registered:

- `SUB-FAR-001` — executable transition integrity;
- `SUB-FAR-002` — deterministic replay continuity;
- `SUB-FAR-003` — bounded model coverage;
- `SUB-FAR-004` — adversarial mutation detection;
- `SUB-FAR-005` — checkpoint and live continuity.

Seven deterministic release contracts cover violation detection,
safety-monotonic hold, replay integrity, bounded exploration, adversarial
mutation detection, checkpoint continuity, and live replay continuity.

`FormalAssuranceEvidenceBundle` binds:

- build identity;
- validation report;
- bounded-model report;
- adversarial report;
- replay records and chain head;
- distinct externally authenticated, hardware-backed Safety Reviewer and
  Formal Methods Reviewer attestations.

Cached reports are recomputed during validation. A bundle cannot pass by
retaining a stale successful verdict after its evidence changes.

## 11. Explicit non-claims

This protocol does not establish:

- correctness of the full continuous plant;
- completeness of the abstract state;
- freedom from compiler, kernel, firmware, or hardware defects;
- cryptographic authenticity of the deterministic digest;
- real-time deadline compliance on deployment hardware;
- regulatory or standards certification;
- safe operation without calibrated sensing and physical qualification;
- proof that every possible adversarial mutation is detected.

## 12. Production qualification still required

Before deployment, the formal layer must be rerun in the complete Rust 1.94
workspace with Clippy and the real HDC/FEP dependencies. Production acceptance
also requires cryptographic replay protection, independent formal review,
hardware-in-the-loop transition faults, power-loss replay testing, clock and
sequence rollover tests, controlled 200 Hz timing measurements, and validation
that the abstraction remains conservative for the actual platform.
