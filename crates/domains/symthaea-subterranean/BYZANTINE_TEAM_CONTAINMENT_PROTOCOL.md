# Byzantine Team Containment Protocol

**Status:** Campaign XXII engineering protocol
**Scope:** `symthaea-subterranean` deterministic team-coordination boundary
**Non-claim:** This crate does not authenticate network identities, provide Byzantine consensus, or prove that a corroborating majority is honest.

## 1. Purpose

Authenticated transport is necessary but insufficient for underground team authority. A peer can possess a valid identity and still be stale, contradictory, compromised, faulty, or operationally dishonest. This protocol prevents one peer—or a split team—from silently converting a claim into movement, rescue, resource, or leadership authority.

The governing rule is:

> Team information may narrow local authority only after identity, freshness, trust, corroboration, and consistency checks. It may never weaken local physical safety, protected return reserve, final-command invariants, actuator isolation, lifecycle restrictions, or decommissioning.

## 2. Trust boundary

The crate consumes `PeerAuthenticationAssertion` records supplied by an external authenticated transport. It checks:

- deployment binding;
- peer identity and local-self exclusion;
- boot epoch and monotonic sequence;
- issuance and expiry steps;
- externally asserted authentication verification;
- externally asserted hardware-backed identity.

It does **not** verify signatures, certificates, attestation chains, secure boot, key custody, or transport confidentiality. False upstream authentication assertions remain outside this crate's threat boundary.

## 3. Peer dispositions

Each known peer has one bounded disposition:

- `ObservationOnly` — reports may be retained as evidence but cannot create team authority;
- `Trusted` — current externally authenticated identity with adequate local behavior score;
- `Restricted` — negative behavior has removed operational authority;
- `Quarantined` — severe or repeated contradictory behavior requires reconciliation.

Behavior events include healthy corroboration, replay attempts, invalid claims, contradictory claims, impossible resource claims, and split-brain participation. Behavior can reduce authority immediately. Positive observations recover score slowly and do not bypass identity expiry.

## 4. High-impact claim quorum

Distress, rescue-route, resource-availability, and leadership claims require reports from distinct trusted peers. Claims are grouped by kind and subject and produce one of four dispositions:

- `NoEvidence`;
- `Insufficient`;
- `Corroborated`;
- `Contradicted`.

Two conflicting value digests are treated as contradiction even when one value has more reporters. This crate deliberately avoids majority-wins behavior for high-impact contradictions.

Claim digests are descriptive identifiers. They are not cryptographic commitments unless an external adapter supplies cryptographically bound values.

## 5. Rescue-claim consistency

A rescue request is checked against retained peer heartbeat evidence and prior requests. Contradictions include materially incompatible depth, battery, capability, or nominal/distress state. Contradictory rescue evidence cannot initiate rescue authority and is retained for reconciliation.

Distress still does not equal consent. Existing rescue handoff requirements remain in force: feasibility, offer, requester acceptance, and explicit begin transition.

## 6. Leadership and split brain

Leadership leases are accepted only from trusted reporters and are replay-resistant by reporter, epoch, and sequence. A current term is classified as:

- `NoLease`;
- `Established`;
- `QuorumLost`;
- `SplitBrain`.

Different leaders or different membership digests in the same highest term constitute split brain. Split brain removes movement in the same control frame; it is not resolved by deterministic tie-breaking.

This is a containment mechanism, not a distributed consensus algorithm.

## 7. Trusted-quorum continuity

Team-dependent actions require a minimum number of fresh trusted participants. Quorum loss escalates through bounded dwell:

- degraded coordination;
- protected return when the local return path remains feasible;
- hold for quorum when return is not feasible;
- immediate hold on split brain.

Local solo operation remains possible when no team dependency exists. The quorum layer does not invent a requirement for peers when the local mission is genuinely independent.

## 8. Resource conservation

Resource offers are bounded by lender identity, borrower identity, expiry, sequence, and finite inventories. Concurrent unresolved offers from one lender cannot exceed that lender's declared energy above its return floor or its finite recovery hardware.

Accepted offers are not treated as locally available. Resources become available only after rendezvous and transfer commitment. This prevents accounting authority from preceding physical transfer.

## 9. Composite containment authority

The composite supervisor produces:

- `Nominal` — trusted team state is internally consistent;
- `ObserveOnly` — evidence may be observed, but team-dependent productive authority is reduced;
- `Reconcile` — contradictory or quarantined state requires bounded reconciliation;
- `ReturnOnly` — persistent trusted-quorum loss requires protected withdrawal;
- `HoldForQuorum` — split brain or unsafe quorum loss removes movement.

The command constraint is safety-monotonic:

- `ObserveOnly` and `Reconcile` tightly bound cutter and auger demand;
- `ReturnOnly` removes productive work and outward motion;
- `HoldForQuorum` removes cutter, auger, tracks, and ballast movement;
- cooling, dewatering, sealant, relay, and roof-support authority remain available when required.

## 10. Authority ordering

The relevant deployed ordering is:

1. local sensing, physical hazards, and protected return truth;
2. resource-conflict and arbitration-recovery authority;
3. distributed cascading-recovery authority;
4. Byzantine team containment;
5. operator, partition, survivability, lifecycle, stewardship, temporal, epistemic, and other established restrictions;
6. actuator isolation;
7. formal transition monitor;
8. final-command invariant monitor;
9. physical actuation.

A partition may remove authority-adding directives such as peer assistance or relay maintenance. It may **not** erase safety-restrictive directives such as tunnel yield, reconciliation hold, quorum return, or quorum hold.

A stricter Byzantine return or hold replaces a weaker team-coordination fallback label so evidence reflects the authority that actually dominated the final command.

## 11. Persistence and evidence

Operational checkpoint schema version 15 preserves:

- peer authentication and trust state;
- claim quorum and replay state;
- rescue consistency state;
- leadership votes and lease assessment;
- trusted-quorum dwell;
- composite containment state;
- resource commitments and existing distributed-recovery state.

Invalid deployment binding, local-agent binding, replay state, or supervisor state is rejected before activation.

Each operational evidence frame records:

- trusted, restricted, and quarantined peer counts;
- contradicted claim groups;
- rescue-claim consistency;
- leadership disposition and selected leader;
- trusted and required quorum participants;
- quorum-loss dwell;
- composite authority and actual command transformation.

## 12. Release contracts

Campaign XXII defines deterministic contracts for:

1. unauthenticated peers having no team authority;
2. contradictory high-impact claims requiring reconciliation;
3. split brain removing movement;
4. lender resource overcommitment being rejected;
5. contradictory rescue requests being rejected;
6. persistent quorum loss selecting protected return;
7. containment preserving emergency recovery actuators;
8. checkpoint restoration preserving quarantine authority.

Release evidence requires distinct hardware-backed Safety Reviewer and Distributed Systems Reviewer identities. Stored verdicts are recomputed from retained evidence.

## 13. Explicit non-claims

This protocol does not establish:

- cryptographic peer authentication;
- Byzantine-fault-tolerant consensus;
- honest-majority assumptions;
- secure time synchronization;
- Sybil resistance;
- correctness of peer sensors or maps;
- safe physical rendezvous or transfer hardware;
- liveness under arbitrary network partitions;
- safety of the complete nonlinear plant.

Those properties require external cryptographic, hardware, networking, HIL, and independent assurance work.
