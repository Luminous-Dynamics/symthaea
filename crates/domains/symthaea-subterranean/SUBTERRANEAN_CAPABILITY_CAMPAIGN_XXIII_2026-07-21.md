# Subterranean Capability Campaign XXIII

**Date:** 2026-07-21
**Theme:** Human-rescue ethics under uncertain team state
**Baseline:** Hardened v22 / cumulative patch 374

## Campaign objective

Campaign XXIII closes the gap between technically feasible rescue and ethically authorized rescue. It gives consent, refusal, withdrawal, emergency intervention, subject-claim consistency, and transparent non-discriminatory triage direct command-level consequences.

The campaign does not attempt medical diagnosis, biometric identity, legal capacity determination, or cryptographic authentication. It consumes bounded externally authenticated assertions and fails closed when authority or evidence is insufficient.

## Delivered capabilities

### Replay-resistant consent continuity

- Case- and subject-bound consent, refusal, and withdrawal.
- Explicit external-authentication assertion.
- Epoch and sequence replay protection.
- Expiry and bounded storage.
- Fresher withdrawal overrides prior handoff acceptance.

### Split-role emergency intervention

- Immediate threat and communication unavailability required.
- Safety Officer and Independent Witness roles.
- Two distinct hardware-backed identities.
- Bounded approval vector and authorization ledger.
- Authorization cannot bypass physical or return safety.

### Opaque subject and care-claim reconciliation

- Opaque identity binding rather than interpreted personal identity.
- Coarse non-diagnostic urgency classes.
- Trusted-reporter boundary.
- Identity and material care conflicts force reconciliation.
- Communication unavailability requires two distinct reporters.

### Transparent role-neutral triage

- Candidate schema excludes social, mission-value, and protected attributes.
- Ranking uses only hazard, survival window, reachability, energy, evidence confidence, consent, and coarse care urgency.
- Refusal and withdrawal always remain ineligible.
- Stable identity appears only as the final deterministic tie-break.

### Same-frame rescue-ethics authority

- `Nominal`.
- `AwaitConsent`.
- `ReconcileClaims`.
- `RescueOnly`.
- `HoldForReview`.

The deployed controller applies this authority before operator control. Rescue-only operation removes productive excavation but retains bounded rescue mobility. Missing or invalid authority removes movement while preserving physical recovery actuators.

### Persistence and evidence

- Operational checkpoint schema version 16.
- Complete rescue-ethics state survives restart.
- Every evidence frame records consent, selected case, emergency authority, claim conflicts, triage invariant, and final command transformation.
- Decision traces and counterfactual explanations identify rescue-ethics blockers.

### Certification integration

Five canonical requirements were added:

- `SUB-HRE-001` — rescue-consent continuity;
- `SUB-HRE-002` — split-role emergency rescue authority;
- `SUB-HRE-003` — rescue-claim reconciliation;
- `SUB-HRE-004` — non-discriminatory transparent triage;
- `SUB-HRE-005` — checkpoint continuity.

Eight deterministic contracts participate in the top-level certification validator. The evidence bundle requires distinct Safety Reviewer and Human Factors Reviewer identities.

## Hardening corrections during review

### Bounded emergency approval vectors

The first authorization structure bounded records but not the number of approvals inside one record. Approval vectors are now limited explicitly.

### Explicit consent authentication

The first consent schema documented authentication as upstream but did not retain an explicit assertion. Consent statements now fail validation unless they are marked externally authenticated.

### Self-consistent evidence validation

Subject-claim assessments now validate reporter counts, identity bindings, communication-unavailability quorum, and bounded reasons. Triage assessments recompute eligible and emergency-authorized counts and verify that the selected subject is actually eligible.

### Borrowing and integration hardening

Triage construction now uses a sequential bounded loop rather than relying on mixed mutable and immutable field borrows inside an iterator closure.

### Authority-state consistency

A `RescueOnly` assessment is invalid unless a selected subject exists and either valid consent or emergency authority supports that selection.

## Release scope

This campaign provides deterministic software constraints and auditable evidence. It does not claim legal, medical, cryptographic, or field qualification. Production release still requires:

- the complete Rust 1.94 workspace;
- Rustfmt, Clippy, and all tests;
- cryptographic identity and consent provenance;
- accessible consent and withdrawal interfaces;
- independent medical, legal, human-factors, and safety review;
- HIL and field rescue exercises;
- protected time and monotonic counters;
- jurisdiction- and community-specific authorization.
