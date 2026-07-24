# Subterranean Capability Campaign XXII

## Byzantine Team Containment and Trusted-Quorum Continuity

**Date:** 2026-07-21
**Incremental range:** patches 351–374
**Baseline:** hardened v21 / cumulative patch 350

## Executive summary

Campaign XXI made distributed rescue, relay contention, resource transfer, and cascading recovery explicit. Campaign XXII addresses the next trust gap: a fresh and authenticated peer may still be faulty, compromised, contradictory, or operationally dishonest.

The campaign adds a bounded team-containment layer that separates authentication from trust, requires corroboration for high-impact claims, detects contradictory rescue and leadership evidence, prevents resource double commitment, and removes team motion when trusted quorum is lost or leadership splits.

The containment system is deliberately safety-monotonic. It may reduce authority to observation, reconciliation, protected return, or hold. It cannot create local movement or weaken physical hazards, return protection, actuator isolation, lifecycle authority, final-command invariants, or terminal decommissioning.

## Delivered capabilities

### 1. Recovery-resource conservation

Concurrent unresolved offers from one lender are accounted together. A lender cannot promise more energy above its protected return floor, dewatering capacity, relay inventory, or roof-support inventory than it declared.

An accepted offer remains unavailable until physical rendezvous and transfer commitment. This prevents ledger state from creating fictional reserves.

### 2. Bounded peer trust registry

Externally authenticated assertions are checked for deployment, identity, epoch, sequence, issuance, expiry, and hardware-backed status. Peers are classified as observation-only, trusted, restricted, or quarantined.

Authenticated identity is not treated as permanent honesty. Replay attempts, invalid claims, contradictory claims, impossible resources, and split-brain participation reduce local authority.

### 3. High-impact claim quorum

Distress, rescue-route, resource-availability, and leadership claims require distinct trusted reporters. Conflicting values produce explicit contradiction instead of majority selection.

### 4. Rescue consistency

Rescue requests are checked against retained heartbeat and prior-request evidence. Material contradictions in depth, battery, capability, or nominal/distress state are rejected and retained for review.

### 5. Leadership split-brain containment

Replay-resistant leadership lease votes track term, proposed leader, membership digest, reporter, epoch, sequence, and expiry. Different leaders or membership views in the same highest term constitute split brain and remove motion immediately.

### 6. Trusted-quorum continuity

Team-dependent action is permitted only while enough fresh trusted participants remain. Bounded quorum-loss dwell escalates from degraded coordination to protected return, or hold when return is infeasible. Split brain holds immediately.

### 7. Composite Byzantine containment

The supervisor emits:

- `Nominal`;
- `ObserveOnly`;
- `Reconcile`;
- `ReturnOnly`;
- `HoldForQuorum`.

The resulting command constraint preserves emergency cooling, dewatering, sealing, relay, and roof-support authority while reducing or removing productive work and motion.

### 8. Runtime authority integration

Byzantine containment is integrated into:

- mission intent;
- motor-safety floors;
- final command transformation;
- decision traces;
- fallback evidence;
- partition interaction;
- operational checkpointing;
- public embodiment APIs.

A partition may clear peer-dependent assistance or relay work, but it cannot erase restrictive yield, reconciliation, quorum-return, or quorum-hold directives.

### 9. Persistence and evidence

Operational checkpoint schema advances to version 15. The checkpoint preserves trust assertions, behavior score, claim quorum, rescue consistency, leadership votes, trusted-quorum dwell, composite authority, and distributed-recovery commitments.

Operational evidence records the exact trust, contradiction, leadership, quorum, and containment state used to transform the final command.

### 10. Certification integration

Six canonical requirements were added:

- `SUB-BYZ-001` — authenticated peer authority boundary;
- `SUB-BYZ-002` — contradictory-claim containment;
- `SUB-BYZ-003` — split-brain containment;
- `SUB-BYZ-004` — resource-claim conservation;
- `SUB-BYZ-005` — trusted-quorum continuity;
- `SUB-BYZ-006` — checkpoint continuity.

The top-level certification gate now executes a Byzantine team-containment validator with eight deterministic contracts.

A self-consistent evidence bundle requires distinct hardware-backed Safety Reviewer and Distributed Systems Reviewer identities. Cached validation is recomputed.

## Cross-layer defects discovered and corrected

### Partition authority erasure

The initial integration cleared every team directive when partition state became non-authoritative. That would also erase safety-restrictive quorum return or hold. The logic now removes only authority-adding peer-dependent directives while preserving restrictive directives.

### Fallback truth downgrade

A weaker cascading-recovery label could remain visible after a stricter Byzantine return or hold had changed the command. The stronger Byzantine fallback now replaces weaker team-coordination labels so evidence matches the actual dominant authority.

### Legacy evidence defaults

New team-evidence fields originally used generic zero/false defaults during deserialization. Legacy evidence could therefore imply an inconsistent rescue claim, empty leadership label, or zero local quorum. Explicit truthful defaults now preserve nominal semantics.

### Restart containment

Additional integration tests verify that conflicting trusted leadership leases produce `HoldForQuorum` and that this restriction survives the distributed-recovery checkpoint path.

## Deterministic release contracts

1. Unauthenticated peer has no team authority.
2. Contradictory trusted claims require reconciliation.
3. Split brain removes movement.
4. Lender overcommitment is rejected.
5. Contradictory rescue request is rejected.
6. Persistent quorum loss selects protected return.
7. Emergency recovery actuators survive containment.
8. Checkpoint restoration preserves quarantine authority.

## Verification scope

The available environment does not contain Cargo, rustc, Rustfmt, Clippy, Nix, or the real workspace path dependencies. Campaign XXII therefore does **not** claim compilation or test execution.

The completed static and reproducibility gates include:

- tree-sitter parsing of every Rust source file;
- unique canonical requirement IDs and codes;
- required traceability links and public exports;
- production forbidden-marker scan;
- `git diff --check`;
- exact incremental patch application over patch 350;
- exact complete-history application over the original imported snapshot;
- tracked path, mode, and blob equality;
- packaged SHA-256 verification.

## Production gates still required

- Complete Rust 1.94 workspace compilation.
- Rustfmt and Clippy with warnings denied.
- Full unit, integration, property, and doctest execution.
- Cryptographic peer authentication and revocation.
- Sybil-resistant enrollment.
- Protected monotonic counters and secure time.
- HIL replay, duplication, reordering, corruption, split-brain, and resource-transfer campaigns.
- Network partition and denial-of-service qualification.
- Independent distributed-systems and safety review.
