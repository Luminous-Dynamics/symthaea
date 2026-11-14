# AOBench — Typed Semantic Continuity

**Status:** Initial deterministic evaluator committed; exact-head Rust qualification pending  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-010 #12167](https://github.com/Luminous-Dynamics/symthaea/issues/12167)  
**Implementation:** `crates/domains/symthaea-organizational-closure/src/semantic_continuity.rs`

## 1. Motivation

The finite labelled-transition checker compares state observations, capabilities, invariants, actions and outcomes. That is a valuable behavioral relation, but action/observation labels cannot safely stand in for all the semantics of a transition.

Two systems may produce the same visible output while differing in:
- which principal acted;
- whether the grant was active and covered the target;
- whether consent was present, current and in scope;
- whether additional irreversible side effects occurred;
- which producer and consumer lineage generated the artifact;
- which external dependencies and substrate resources were used.

AOBench therefore evaluates typed semantic continuity separately from finite behavioral continuity. These planes remain non-compensatory.

## 2. Implemented data model

The first Rust increment adds `semantic_continuity.rs` and exports it from the nested package root.

Each `SemanticTransition` contains:

| Plane | Typed evidence |
|---|---|
| Authority | Principal, trust-root digest, grant identity, exact action/target scopes, grant status and evidence reference |
| Consent | Subject, scope, status and evidence reference |
| Effects | Complete-inventory disposition, effect ID/class/target, reversibility and evidence reference |
| Provenance | Input/output digest links, producer/consumer identities and independent receipts |
| Substrate | Dependency identity, observed availability/denial state and receipt reference |

The transition key is an explicit action/target/outcome tuple. State and component IDs are not used as a proxy for semantic equivalence.

## 3. Comparison rules

### Authority

Both records must have active grants and qualified evidence. Principal, trust root, action scopes and target scopes must match exactly in v0.1. Grant IDs may rotate during renewal, provided the semantics remain the same.

A change of principal or trust root is rejected until an explicit externally authorized migration protocol is separately qualified.

### Consent

An action can declare that consent is mandatory. Missing, revoked, out-of-scope or unresolved consent cannot pass. Consent IDs may rotate when renewed, but subject and scope must remain the same.

### Effects

Both transitions require an independently complete effect inventory. Baseline effects must remain represented. Candidate-added irreversible effects fail closed in v0.1. Additional reversible effects are admitted only by an explicit frozen allowlist.

Unknown reversibility is `Inconclusive`, not a pass.

### Provenance

Lineage steps must have producer and consumer identities, independently qualified receipts and digest continuity between adjacent links.

The candidate chain may differ from the baseline chain. This is intentional: a legitimate producer replacement should not fail merely because its internal component IDs or artifact digests differ. Both chains must nevertheless be independently reconstructible.

### Substrate

External dependency sets and observed dependency statuses must match exactly in v0.1. A change is surfaced as a substrate-continuity failure; it is not hidden inside a transition label.

## 4. Relation direction

- **Simulation:** every baseline semantic transition must be present in the candidate. Candidate-only transitions are counted and independently validated but are not part of the one-way baseline-preservation claim.
- **Bisimulation:** candidate-only semantic transition keys are rejected as well.

The typed evaluator complements the LTS relation. It is not a replacement for the state-pair relation or its backward propagation of delayed divergence.

## 5. Evidence and claim boundary

An `EvidenceRef` carries an independently supplied verifier disposition, digest and reference. The module validates presence and disposition; it does not check signatures, signer authorization, oracle truth, physical execution or effect-inventory completeness.

A caller-controlled `Verified` enum is not a security boundary. Production adapters must bind these values to independently authenticated #12014 receipts, #12067 intervention evidence, and external authority/consent systems.

Disposition rules:
- explicitly rejected evidence or violated contract → `NotPreserved`;
- unavailable evidence, unresolved status or exhausted comparison budget → `Inconclusive`;
- all mandatory planes preserved → `Preserved`.

The report retains separate plane qualifications. No weighted average is produced.

## 6. Deterministic challenge matrix

The initial test module covers:
- same outputs with a different principal;
- grant and consent receipt renewal without semantic scope change;
- revoked and missing required consent;
- added hidden irreversible effects;
- explicitly allowlisted reversible effect additions;
- different but joined and receipted lineage;
- broken lineage links;
- unavailable evidence;
- changed substrate dependencies;
- simulation versus bisimulation candidate-only transitions;
- exhausted comparison budget;
- missing action authority scope.

AOBENCH-009 #12140 should mutate each predicate and retain a condition-removed positive twin. The present unit tests are not yet a mutation campaign.

## 7. Next qualification work

1. Exact-head format, compile, unit tests and Clippy on this module.
2. Mutation operators for authority, consent, effects, lineage and substrate predicates.
3. Signed receipt adapter that supplies trusted evidence dispositions.
4. Externally authorized trust-root and principal migration protocol.
5. Explicit effect inventory observer and intervention/effect correspondence from #12067.
6. Stochastic approximate semantic relations only after #12048 assay adequacy is qualified.
7. End-to-end replay on a qualified organizational environment.

## 8. Claim boundary

The implementation is a deterministic evaluator over supplied typed transition envelopes. It does not establish that the envelope describes every physical effect, authenticate evidence itself, prove universal behavioral equivalence, or establish biological autopoiesis, life, sentience or consciousness.

## References

- [Operational continuity specification #12121](https://github.com/Luminous-Dynamics/symthaea/issues/12121)
- [Mutation adequacy #12140](https://github.com/Luminous-Dynamics/symthaea/issues/12140)
- [Authenticated runtime evidence #12014](https://github.com/Luminous-Dynamics/symthaea/issues/12014)
- [Intervention integrity #12067](https://github.com/Luminous-Dynamics/symthaea/issues/12067)
- [ACED-Bench](https://arxiv.org/abs/2610.14381)
- [CausalArena](https://arxiv.org/abs/2609.11897)
