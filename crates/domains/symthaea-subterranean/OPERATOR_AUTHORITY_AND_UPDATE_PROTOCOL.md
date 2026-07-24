# Subterranean Operator Authority and Update Protocol

Date: 2026-07-20

## Purpose

This protocol defines how externally authenticated human commands, degraded-operation state, software/configuration activation, audit continuity, and restart recovery interact with the subterranean platform.

It is deliberately narrower than a transport-security protocol. The crate does **not** verify digital signatures, enroll operators, install software bytes, manage secure boot keys, or claim that its deterministic test digests are cryptographically collision resistant.

## Authority ordering

The deployed authority order is:

1. Physical hazard assessment and verified recovery planning.
2. Degraded-operation supervisor and recovery lock.
3. Restrictive operator constraints.
4. Team right-of-way and explicitly accepted rescue work.
5. Mission executive and logistics admission.
6. Learned nominal controller output.

A lower layer cannot weaken a higher layer. An operator may stop, hold, request return, enter maintenance, or select a nominal mission, but cannot command through a physical hazard or clear a recovery lock by sending a single resume message.

## Operator command boundary

`OperatorCommandEnvelope` carries:

- stable operator identity;
- declared role;
- upstream authentication level;
- epoch and monotonic sequence;
- proposal identifier;
- issue and expiry steps;
- command.

`OperatorTrustPolicy` independently validates role, authentication strength, freshness, future skew, lifetime, and quorum metadata. `OperatorAuthority` then rejects stale epochs and replayed sequences.

### Authentication levels

- `Unverified`: never receives actuation authority.
- `TransportAuthenticated`: may issue ordinary restrictive or mission commands when the role permits.
- `HardwareBacked`: required for recovery/resume approvals.

These values are assertions from an upstream security boundary. They are not created by signature verification inside this crate.

## Recovery quorum

`ResumeNominal` requires at least two distinct hardware-backed approvals from supervisor/safety roles, sharing one nonzero proposal identifier and one expiry boundary. Physical hazards must already be clear.

One operator cannot clear an emergency stop, hold, maintenance lock, or degraded recovery lock alone.

## Safety-monotonic command application

Operator constraints are applied to the learned nominal command **before** physical recovery planning:

- Emergency stop, hold, and maintenance lock remove cutter, auger, track, ballast, and recovery-actuator authority while preserving temperature-dependent cooling.
- Return-home removes cutting and forces conservative inbound motion.
- Mission selection changes intent but does not bypass hazard, team, logistics, or maintenance gates.

Physical recovery planning remains downstream and may add cooling, dewatering, sealing, roof support, relay deployment, or withdrawal needed to preserve the platform.

## Degraded-operation supervision

The supervisor observes:

- operator-link freshness;
- control-loop watchdog health;
- checkpoint validity;
- reboot count in a bounded window;
- battery and return feasibility;
- whether the platform is at the surface or a service bay.

Modes are:

- `Normal`
- `OperatorLinkLost`
- `AutonomousReturn`
- `SafeHold`
- `RecoveryRequired`

Link loss uses a grace period. After the grace period, a feasible and funded return selects autonomous return; otherwise the platform holds. Repeated watchdog failures, reboot loops, or invalid checkpoint state latch `RecoveryRequired` at Red safety. Link restoration alone cannot clear this state.

## Update activation boundary

`UpdateManager` does not fetch, verify, unpack, or install bytes. It controls whether an externally verified artifact may become active.

Staging requires:

- supported manifest schema;
- nonzero externally supplied artifact/configuration/rollback digests;
- a newer epoch;
- checkpoint compatibility;
- surface or service-bay location;
- no active work;
- clear physical hazards;
- at least 40% battery;
- active maintenance lock.

Activation enters `PendingHealth` for a bounded window. A failed or late health check requires rollback to the previous digest. Successful health validation commits the activation.

## Audit integrity chain

`AuditLedger` forms a bounded previous-digest chain over operator and update events. `AuditDigestProvider` is pluggable so production can use the project’s cryptographic trust fabric.

`DeterministicAuditDigest` exists only for reproducible tests and continuity comparison. It is not a signature and must not be used as an adversarial tamper-proofing claim.

## Restart recovery

Operational checkpoint schema v2 persists:

- learned controller state;
- mission graph, scheduler, logistics, and maintenance;
- operator replay state and active constraint;
- degraded-operation state;
- staged or pending update state.

Schema v1 remains readable through defaulted authority fields.

`RecoveryJournal` uses two alternating generations. Each slot carries an externally pluggable integrity digest. Recovery selects the newest valid generation and falls back to the older slot when the newest is corrupt or unverifiable.

## Evidence and acceptance gates

Every runtime evidence frame can now include operator constraint, accepted/rejected command counts, degraded mode, link-loss duration, update state, successful activations, and rollbacks.

`AuthorityValidator` gates:

1. replay resistance;
2. independent recovery quorum;
3. physical-hazard resume blocking;
4. audit-chain continuity and modification detection;
5. failed-update rollback;
6. watchdog recovery lock;
7. two-slot journal fallback.

## Explicit non-claims

This campaign does not establish:

- cryptographic operator identity;
- secure transport;
- key enrollment or revocation;
- secure boot or measured boot;
- artifact signature verification;
- atomic filesystem installation;
- hardware watchdog correctness;
- field certification or functional-safety compliance.

Those concerns must be integrated and validated at the full-system boundary.
