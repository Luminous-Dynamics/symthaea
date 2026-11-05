# ENG-ARTIFACT-TRUST-001 — engineering artifact authenticity and provenance benchmark

**Tracking issue:** [#10647](https://github.com/Luminous-Dynamics/symthaea/issues/10647)  
**Status:** research specification; synthetic blueprints only; no candidate or verifier score  
**Authority:** evaluation-only. No production credentials, trust roots, supplier approvals, evidence admission, procurement, fabrication or release authority.

## 1. Evaluation thesis

The benchmark evaluates whether an engineering system can distinguish:

```
digest consistency
!= authenticated producer

valid signature
!= authorized signer for this claim and subject

authenticated provenance
!= semantic correctness

semantic correctness
!= evidence applicability/currentness

evidence applicability
!= engineering approval or release
```

A package can be self-consistent and entirely attacker-authored. A valid signature can come from a signer that is not trusted for the predicate or artifact class. A correctly signed artifact can still be replayed against the wrong generation or article.

This benchmark is separate from MBD-INTEROP-001: that programme measures semantic preservation across exchange; this one measures authenticity, scope, freshness and chain integrity.

## 2. Ownership and standards to compose

- [ENG-REA-008 #8135](https://github.com/Luminous-Dynamics/symthaea/issues/8135): authenticated artifact receipt and trust-policy direction.
- [CEB-TASK-CUSTODY-001 #10436](https://github.com/Luminous-Dynamics/symthaea/issues/10436): separation of agent-visible inputs and evaluator-only trust material.
- [ETK #1738/#1730/#1760](https://github.com/Luminous-Dynamics/symthaea/issues/1738): evidence admission, applicability/currentness and authority.
- [ROB-CATALOG #4877](https://github.com/Luminous-Dynamics/symthaea/issues/4877): physical component and source-document identity.
- [ROB-REALIZE #4859](https://github.com/Luminous-Dynamics/symthaea/issues/4859): design, fabrication, article and metrology lineage.
- [MBD-INTEROP-001 #10595](https://github.com/Luminous-Dynamics/symthaea/issues/10595): product-definition, quality and manufacturing data exchange.
- [ENG-DESIGN-CHANGE-QUAL #10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543): claim/evidence applicability after change.

Do not create a second signing library, trust-root registry, artifact database, evidence/currentness engine or supply-chain ledger. The benchmark defines fixtures, expected dispositions and scorer integrity.

External profiles:
- [in-toto Attestation Framework](https://github.com/in-toto/attestation)
- [DSSE envelope specification](https://github.com/in-toto/attestation/blob/main/spec/v1/envelope.md)
- [SLSA Build Provenance v1.1](https://slsa.dev/spec/v1.1/provenance)
- [SLSA artifact verification guidance](https://slsa.dev/spec/v1.1/verifying-artifacts)

These are references for evaluation profiles. No in-toto conformance or SLSA level is claimed.

## 3. Trust model

A verifier receives:
1. An artifact bundle and claimed subject digests.
2. One or more signed statements and provenance links.
3. An independently configured trust policy and trust roots.
4. Expected repository/organization/workflow identity and permitted predicate scope.
5. Target design/configuration/article generation and freshness policy.
6. A verifier implementation and profile whose own identity is pinned.

**The candidate-controlled bundle must never supply the only trust root, expected signer identity, authorization scope or revocation policy used to verify itself.**

Use synthetic keys and test roots only. Do not add real private keys, production secrets, supplier credentials or customer data to the public corpus.

## 4. Fixture families

### Positive controls

- **TRUST-P01 — signed design artifact:** valid synthetic signature, expected producer/workflow, exact source revision and geometry digest.
- **TRUST-P02 — signed inspection package:** valid signature bound to exact as-built article, lot, characteristic, instrument/calibration profile and inspection event.
- **TRUST-P03 — complete provenance chain:** design export → manufacturing plan → execution record → article identity → inspection result, with correct subject linkage and policy.

### Critical negative controls

- **TRUST-N01 — self-consistent whole-bundle replacement:** replace artifact and unsigned manifest/receipt together; content hashes still match but no trusted signature authenticates the new subject.
- **TRUST-N02 — untrusted signer:** cryptographic verification succeeds, but the signer is absent from the verifier's independent trust root.
- **TRUST-N03 — signer scope mismatch:** signer is trusted for one predicate/artifact class but not the engineering assertion being made.
- **TRUST-N04 — stale-generation replay:** valid prior-generation signature is replayed against a newer design or configuration.
- **TRUST-N05 — subject substitution:** signature is valid but binds to another component variant, article, lot, feature, characteristic or inspection event.
- **TRUST-N06 — provenance-chain splice:** one or more required transformation, builder, process or inspection links are missing/replaced.
- **TRUST-N07 — trust-root substitution:** package supplies a new public key or trust policy and claims it as authoritative.
- **TRUST-N08 — profile downgrade:** weaker/unsupported schema or predicate profile is silently accepted as equivalent.
- **TRUST-N09 — revoked/expired credential:** cryptographic signature verifies, but the independently configured policy rejects it.
- **TRUST-N10 — timestamp/timebase manipulation:** old telemetry, calibration or inspection evidence is made to appear current through altered time claims.
- **TRUST-N11 — claim-tier escalation:** integrity-only or build-provenance evidence is presented as engineering verification, inspection approval or release authority.
- **TRUST-N12 — cross-organization identity collision:** a same-named repository, supplier, machine, component or article is rebound to a different organization/tenant identity.

Each attack needs one or more benign near-miss controls. A verifier that rejects every nontrivial package is not useful.

## 5. Scoring contract

Report separately:
- Critical false accepts (non-compensatory).
- Correct valid-package acceptance.
- Untrusted/revoked/out-of-scope signer rejection.
- Subject-binding and source-generation mismatch rejection.
- Replay/freshness correctness.
- Provenance-chain completeness.
- Trust-root independence.
- Correct Unknown/Unsupported handling for unresolved revocation or profile state.
- False-rejection rate on valid near-miss controls.
- Claim-tier escalation count.
- Offline verification, latency and resource cost.
- Full attempted/eligible/graded/unknown/unsupported/ungraded/aborted denominators.

Never average a critical false acceptance into a composite score.

## 6. Evaluator integrity

- Candidate and evaluator trust policies are separate objects with separate digests and access controls.
- Candidate process cannot modify or read evaluator-only trust roots, attack labels or hidden expected dispositions.
- Public development cases and future held-out cases are split by signer, trust-policy family, subject lineage and attack family—not by random row.
- Include known-good, empty/missing-attestation, wrong-scope, wrong-subject and corrupted-verifier controls.
- Use independent verifier implementations where feasible. Do not let a signer or verifier certify its own behavior.
- Missing revocation state or unsupported policy semantics must be Unknown/Unsupported, not Pass.
- A valid signature may support authenticity only; it cannot mint ETK admission, professional approval or release.

## 7. Initial acceptance gates (provisional)

- Zero critical false accepts on the frozen synthetic attack corpus.
- 100% correct rejection of wrong subject, untrusted signer, scope mismatch, trust-root substitution and stale-generation replay controls.
- 100% correct classification of valid positive controls under the exact frozen policy.
- Zero claim-tier escalations.
- All missing/unsupported trust facts remain explicit Unknown/Unsupported.
- Deterministic replay of verifier outputs for the same subject, policy and implementation.

Do not freeze operational false-accept claims from a small synthetic corpus. Expand independent attack-family and key/identity/organization holdouts before reliability claims.

## 8. Staging

- **M0:** threat model, trust policy and claim-tier review.
- **M1:** synthetic positive/negative fixture materialization.
- **M2:** independent verifier and mutation tests.
- **M3:** matched verifier baselines and key/identity/family-held-out attacks.
- **M4:** cross-project integration with AP242/QIF/MTConnect artifact chains.
- **M5:** bounded pilot with explicit organization, supplier, custody, revocation and security approvals.

## 9. Claim ceiling

A passing result establishes only that the named verifier and trust-policy profile correctly handled the exact synthetic cases. It does not establish universal supply-chain security, trust in any real organization, evidence admission, product safety, professional engineering approval, procurement, fabrication, regulatory compliance or release authority.
