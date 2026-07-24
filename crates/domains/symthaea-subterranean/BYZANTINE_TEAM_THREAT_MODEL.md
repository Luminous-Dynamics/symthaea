# Byzantine Team Threat Model

## Assets protected

- Local command authority and final physical actuator command.
- Protected return energy and recovery reserves.
- Rescue consent and rescue feasibility decisions.
- Team leadership and membership interpretation.
- Shared route, occupancy, relay, and resource evidence.
- Checkpoint continuity and operational audit records.

## Adversary and failure classes

### Unauthenticated or weakly authenticated peer

A message names a valid-looking agent but lacks externally verified and hardware-backed identity. The report may be retained as observation-only evidence but cannot create team authority.

### Replay or stale identity

A previously valid assertion, heartbeat, claim, lease, or offer is replayed. Epoch and sequence state rejects non-monotonic records; expiry removes authority.

### Authenticated but faulty peer

A peer may be correctly authenticated yet report impossible resources, inconsistent rescue state, contradictory high-impact claims, or conflicting leadership. Authentication does not imply operational correctness.

### Resource double commitment

One lender offers the same finite reserve to multiple borrowers. The resource ledger accounts for concurrent unresolved commitments and rejects offers that would violate the lender's declared return floor or finite inventory.

### Contradictory rescue claim

A rescue request conflicts materially with retained heartbeat or prior request evidence. The request is rejected or contained for reconciliation; distress alone does not redirect another machine.

### Split-brain leadership

Trusted reporters assert different leaders or membership views in the same term. Motion is removed immediately. No deterministic winner is chosen inside this crate.

### Quorum erosion

Team-dependent work continues while trusted peers expire, become restricted, or disappear. The trusted-quorum supervisor degrades coordination, then selects protected return or hold.

### Partition interaction

A communications partition could otherwise clear all team directives. Authority-adding directives are removed when peer state is not authoritative, but safety-restrictive directives remain active.

### Evidence downgrade

A stronger hold could be reported as a weaker coordination state. Fallback selection now permits Byzantine return or hold to replace weaker team labels so the retained explanation matches the final command.

## Trust assumptions

The crate assumes:

1. The external adapter truthfully sets authentication and hardware-backed flags.
2. Deployment and agent identifiers are provisioned correctly.
3. Local monotonic step progression is protected by the existing temporal and formal assurance layers.
4. Local sensor, hazard, return, actuator, and invariant authorities remain independent of peer claims.
5. A trusted peer can still fail or lie; local trust may therefore decrease after authenticated admission.

## Safety properties

- No unauthenticated peer can create team authority.
- A single trusted peer cannot unilaterally establish a high-impact corroborated claim.
- Contradictory trusted claims narrow authority rather than selecting a winner.
- Split brain removes movement in the same control frame.
- Quorum loss cannot silently leave team-dependent motion nominal.
- Resource offers cannot exceed declared lender reserves across simultaneous commitments.
- Accepted-but-untransferred resources are not locally spendable.
- Containment preserves physically necessary recovery actuators.
- Restart cannot erase persisted trust, contradiction, leadership, quorum, or containment state.

## Liveness limits

Containment may intentionally sacrifice team mission progress. In particular:

- split brain may hold indefinitely until externally reconciled;
- no safe return plus lost quorum results in hold;
- contradictory claims are not resolved by majority vote;
- quarantined peers cannot regain authority merely through continued messaging;
- partition recovery does not guarantee eventual connectivity.

The separate arbitration-recovery and distributed-recovery layers address bounded progress where possible, but this crate does not promise liveness under arbitrary Byzantine behavior.

## Required external mitigations

- Cryptographic identity and message authentication.
- Protected monotonic counters and secure boot epochs.
- Sybil-resistant enrollment and revocation.
- Authenticated time or trusted monotonic hardware clocks.
- Independent sensor and map corroboration.
- Physical transfer interlocks and metering.
- Network-level partition and denial-of-service defenses.
- HIL campaigns with packet loss, duplication, delay, reordering, corruption, and compromised peers.
- Independent distributed-systems and safety review.
