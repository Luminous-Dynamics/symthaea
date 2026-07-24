# Subterranean Capability Campaign XVI

## Formal Runtime Assurance and Adversarial Validation

**Date:** 2026-07-20  
**Incremental range:** patches 206–231  
**Cumulative range:** patches 1–231  
**Baseline:** Capability Campaign XV / patch 205

## Executive summary

Campaign XVI adds an independent formal-assurance layer to the deployed command
path. It reduces the platform to a finite authority abstraction, checks every
abstract transition, removes movement in the same control frame when that
contract is violated, records a deterministic replay chain, exhaustively
explores preregistered bounded properties, and attacks its own retained evidence
with a deterministic mutation corpus.

The result is not a proof of the full nonlinear system. It is a deliberately
bounded and executable assurance argument whose authority is narrower than the
existing physical safety layer and whose failures are visible in the same
operational, accountability, checkpoint, traceability, and certification paths
as other safety interventions.

## Patch campaign

### 206. Bounded runtime-state abstraction

Introduces a schema-versioned `FormalRuntimeState` containing the minimum
runtime facts needed to check command authority: step, safety, hazards, return
feasibility, capability, operator constraint, policy authority, terminal asset
state, partition authority, service location, and command.

### 207. Executable transition contract

Adds structured transition violations for malformed state, command bounds,
sequence regression and gaps, missing hazard escalation, infeasible-return work,
hold motion, unauthorized candidate use, terminal-state reversal, partition
motion, and operator-constraint contradictions.

### 208. Latched formal hold

Adds a safety-monotonic runtime monitor. Any violation removes work and motion
in the same frame while preserving containment actuators. Clearance requires a
clean dwell at a safe service location.

### 209. Chained deterministic replay ledger

Adds bounded replay frames, sequence continuity, previous-digest linkage, chain
head verification, frame validation, and structured replay failures.

### 210. Composite formal-assurance supervisor

Composes the transition monitor and replay ledger behind one validated runtime
boundary.

### 211–212. Checkpoint persistence and embodiment lifecycle

Advances operational checkpoint schema to version 9 and persists the formal
supervisor. Invalid formal monitor or replay state fails before live activation.

### 213. Live pre-invariant enforcement

Places formal transition enforcement after physical actuator isolation and
before the final-command invariant monitor. Formal violations escalate safety,
record `FormalAssuranceHold`, and alter the command before plant motion.

### 214. Accountability integration

Adds the formal assurance authority stage to command-decision traces and
counterfactual explanations.

### 215. Operational evidence integration

Records formal hold state, violation count, latest violations, replay-record
count, chain validity, and append failures in bounded runtime evidence and
summaries.

### 216. Exhaustive bounded model checker

Checks six finite properties over 32 preregistered abstract cases:

- hazard escalation;
- protected return reserve;
- hold immobility;
- candidate authority;
- terminal decommissioning;
- contiguous runtime sequence.

### 217. External replay verification boundary

Exposes deterministic replay verification for evidence and reviewer tooling
without exposing mutation of live supervisor internals.

### 218–219. Adversarial mutation corpus and warning closure

Adds eight deterministic evidence and transition mutations and requires every
one to be detected. Keeps the validator warning-clean under `-D warnings`.

### 220. Overflow-safe replay mixing

Full-suite testing found that ordinary multiplication in the deterministic
non-cryptographic digest could panic in debug builds. Mixing now uses explicit
wrapping arithmetic.

### 221. Bounded sequence-edge correction

Model exploration found an incorrect expectation near saturating `u64` step
arithmetic. The checker now treats the bounded terminal edge consistently and
does not manufacture a false sequence gap.

### 222. Deployment-facing assurance API

Exposes formal snapshot and replay evidence through the embodiment boundary for
qualification and independent review.

### 223. Release-blocking formal contracts

Adds seven deterministic acceptance contracts covering transition detection,
hold behavior, replay integrity, bounded exploration, mutation detection,
checkpoint continuity, and live replay continuity.

### 224–226. Requirements, traceability, and certification

Registers five release-applicable requirements:

- `SUB-FAR-001` — formal transition integrity;
- `SUB-FAR-002` — replay continuity;
- `SUB-FAR-003` — bounded model coverage;
- `SUB-FAR-004` — adversarial mutation detection;
- `SUB-FAR-005` — checkpoint and live continuity.

Links them to tests, scenarios, monitors, evidence, and the top-level
certification validator.

### 227–228. Self-consistent formal evidence bundle

Binds build identity, recomputed validation reports, model-check report,
adversarial report, replay records, chain head, and two distinct externally
authenticated hardware-backed reviewer roles. Uses the workspace JSON API and a
pluggable digest boundary.

### 229. Formal Runtime Assurance Protocol

Documents authority ordering, abstraction limits, transition semantics, replay,
model checking, adversarial scope, release evidence, non-claims, and remaining
production qualification.

### 230. Campaign record

Publishes this ordered implementation account.

### 231. Verification record

Publishes static, executable, and clean-room reconstruction evidence.

## Causal authority result

The deployed suffix is now:

```text
constrained nominal command
  -> actuator isolation
  -> formal transition monitor
  -> final-command invariant monitor
  -> plant
  -> deterministic replay evidence
```

The formal monitor can only remove authority. It cannot restore a failed
actuator, bypass a physical hazard, make a return path feasible, reactivate a
retired policy, reverse decommissioning, or grant candidate-policy authority.

## Validation result

Offline compatibility-workspace validation produced:

- warning-denied `cargo check --all-targets`: pass;
- 350 tests passed;
- 0 tests failed;
- 1 controlled-hardware timing benchmark intentionally ignored;
- all seven formal release contracts passed;
- all six bounded properties passed over 32 explored cases;
- all eight preregistered mutations were detected.

The complete clean-room and packaging evidence is recorded separately in
`SUBTERRANEAN_CAPABILITY_CAMPAIGN_XVI_VERIFICATION.md`.

## Remaining qualification boundary

Campaign XVI does not convert the crate into a formally verified physical
system. Production acceptance still requires the complete Rust 1.94 workspace,
Clippy, an approved cryptographic replay provider, independent formal review,
hardware-in-the-loop transition corruption, power-loss replay testing, step and
clock rollover tests, controlled timing evidence, and validation that the
finite abstraction conservatively represents the actual machine and site.
