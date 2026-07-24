# Human Rescue Ethics Protocol

**Status:** Campaign XXIII engineering protocol
**Scope:** `symthaea-subterranean` rescue-consent, triage, and team-authority boundary
**Non-claim:** This crate does not identify people, diagnose medical conditions, determine legal capacity, verify signatures, or replace trained human rescue and medical judgment.

## 1. Purpose

Underground rescue creates a dangerous authority asymmetry: distress, urgency, uncertain identity, and incomplete communications can pressure an autonomous platform to move before consent, evidence, and safe feasibility are established. This protocol makes those conflicts explicit and safety-monotonic.

The governing rule is:

> Rescue urgency may prioritize among ethically eligible cases, but it may not manufacture consent, resolve contradictory identity or care claims by majority vote, discriminate by social or protected attributes, or bypass physical safety and protected return truth.

## 2. Authority boundary

The crate consumes externally authenticated consent statements and hardware-backed emergency approvals. It checks schema, case and subject binding, epoch, monotonic sequence, issuance, expiry, and bounded record counts.

It does **not** verify:

- digital signatures or certificate chains;
- biometric identity;
- legal competence or incapacity;
- medical diagnoses;
- secure time or protected counters;
- authenticity of upstream hardware-attestation claims.

A false upstream authentication assertion remains outside this crate's trust boundary.

## 3. Case-specific consent continuity

A rescue-consent statement is bound to one subject and one rescue case. It may state:

- `Consent`;
- `Refuse`;
- `Withdraw`.

Consent records are replay-resistant by epoch and sequence and expire at a bounded step. A fresher refusal or withdrawal overrides an earlier acceptance. Distress, silence, role, urgency, and peer opinion never imply consent.

An accepted rescue handoff counts as case-specific consent only while no fresher authenticated refusal or withdrawal exists.

## 4. Emergency intervention authority

When the subject cannot communicate and faces an immediate threat, rescue may proceed only with:

- a current case-bound authorization;
- an explicit immediate-threat assertion;
- corroborated communication unavailability;
- two distinct hardware-backed operators;
- both `SafetyOfficer` and `IndependentWitness` roles;
- bounded issuance and expiry;
- replay-resistant epoch and sequence.

Emergency authority is not a general override. It applies only to the named subject and rescue case and does not weaken route feasibility, return reserve, physical hazards, final-command invariants, actuator isolation, decommissioning, or environmental restrictions.

## 5. Opaque identity and care claims

Rescue-subject claims contain an opaque externally generated identity binding. The crate does not interpret that binding as a name, biometric, legal identity, or proof of personhood.

Trusted claims may provide only coarse care urgency:

- `Unknown`;
- `Stable`;
- `Urgent`;
- `Critical`.

The ledger detects:

- multiple subject identifiers for one case;
- multiple opaque identity bindings;
- materially contradictory care urgency;
- insufficient independent corroboration of communication unavailability.

Contradiction produces reconciliation or hold authority. The crate does not select the largest voting bloc as truth.

## 6. Transparent non-discriminatory triage

The triage candidate schema intentionally excludes occupation, organizational role, nationality, wealth, payload value, mission value, race, religion, sex, disability, age, and other protected or socially ranked attributes.

Allowed ranking factors are limited to:

- physical hazard severity;
- bounded survival window;
- route reachability;
- rescue energy cost;
- evidence confidence;
- case-specific consent;
- coarse corroborated care urgency.

Stable `AgentId` is used only as the final deterministic tie-break after every allowed factor is equal.

Refusal or withdrawal is ineligible regardless of urgency. Identity or care contradiction is ineligible pending reconciliation. An emergency-authorized subject is eligible only when the emergency authority and communication-unavailability evidence are both valid.

## 7. Composite rescue-ethics authority

The supervisor emits:

- `Nominal` — no rescue-ethics restriction is active;
- `AwaitConsent` — rescue motion awaits case-specific consent or valid emergency authority;
- `ReconcileClaims` — trusted subject or care claims conflict;
- `RescueOnly` — valid rescue authority exists; productive excavation is removed while rescue mobility remains;
- `HoldForReview` — active rescue authority has become ethically invalid or the non-discrimination invariant failed.

The command transformation is safety-monotonic:

- `RescueOnly` removes cutter and auger demand;
- `AwaitConsent`, `ReconcileClaims`, and `HoldForReview` remove cutter, auger, tracks, and ballast motion;
- cooling, dewatering, sealing, relay deployment, and roof support remain available when physically required.

## 8. Team integration

The team coordinator combines:

- explicit rescue handoff state;
- consent continuity;
- emergency authorization;
- opaque subject claims;
- transparent triage;
- existing Byzantine containment and distributed-recovery authority.

New team directives are:

- `AwaitRescueConsent`;
- `ReconcileRescueClaims`;
- `HoldForRescueReview`.

Byzantine split-brain or quorum holds remain higher-priority restrictions. Rescue ethics cannot turn an untrusted or contradictory team plan into movement.

## 9. Deployed authority ordering

The relevant ordering is:

1. fused local sensing and physical hazards;
2. protected return and resource-conflict authority;
3. cascading distributed recovery;
4. Byzantine team containment;
5. human-rescue ethics;
6. operator and partition constraints;
7. survivability, lifecycle, stewardship, epistemic, and temporal restrictions;
8. actuator isolation;
9. formal transition monitor;
10. final-command invariant monitor;
11. physical actuation.

Every later layer may preserve or remove authority. No rescue-ethics layer may recreate authority removed earlier.

## 10. Persistence and evidence

Operational checkpoint schema version 16 preserves:

- consent records and replay state;
- emergency authorizations;
- subject-claim evidence;
- triage policy and last assessment;
- composite rescue-ethics authority;
- the existing rescue handoff and distributed-recovery state.

Invalid or internally inconsistent rescue-ethics state is rejected before checkpoint activation.

Each operational evidence frame records:

- rescue-ethics authority;
- consent disposition;
- selected subject and case;
- emergency-authorization use;
- identity and care conflicts;
- non-discrimination invariant status;
- triage candidate count;
- the actual command transformation and fallback label.

## 11. Release contracts

Campaign XXIII defines deterministic contracts for:

1. consent replay rejection;
2. withdrawal stopping an active rescue;
3. emergency intervention requiring two independent roles;
4. conflicting identity claims requiring reconciliation;
5. refusal dominating urgency;
6. triage structurally excluding protected attributes;
7. recovery actuators surviving an ethics hold;
8. checkpoint restoration preserving consent authority.

Release evidence requires distinct externally authenticated, hardware-backed Safety Reviewer and Human Factors Reviewer identities. Stored validation is recomputed.

## 12. Explicit non-claims

This protocol does not establish:

- medical diagnosis or clinical triage validity;
- legal consent, competence, guardianship, or emergency-treatment authority;
- biometric identity or anti-impersonation guarantees;
- cryptographic authentication;
- statistical fairness across real populations;
- accessibility adequacy;
- safe physical extraction or transport;
- correctness of the nonlinear plant;
- regulatory or community authorization.

Those properties require external legal, medical, human-factors, cryptographic, hardware, HIL, and independent review.
