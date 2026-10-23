# Engineering Trust Kernel — Authority Bypass Census V1

**Status:** source audit / migration inventory, not closure evidence  
**Reference main:** `3afeee3d40af0bae0b85e70869571e024c28f07b`  
**ETK production bridge:** draft PR #1730

## Purpose

The Engineering Trust Kernel is not complete merely because a typed admission API exists. This census records every currently identified path in the engineering/formal-safety facade that can create, consume, or amplify `Discharged` authority without crossing the new ETK admission/receipt theorem.

The central migration invariant is:

```text
computation happened
!= assessment passed
!= candidate evidence
!= admitted evidence
!= discharge receipt
!= present-tense discharged obligation
!= qualified design
!= deployment/fabrication/actuation authority
```

Until every authority-bearing consumer is migrated, #1730 must not be described as closing the engineering authority boundary globally.

## A. Direct discharge creation bypasses

### A1. Generic external-simulation fan-out

`EngineeringManager::evaluate_concept()` runs each `SimulationRequest`; when any result has `result.converged == true`, it currently marks **every** safety-case obligation whose `expected_evidence == EvidenceKind::Simulation` as `ObligationStatus::Discharged`.

Problems:

- no exact obligation ↔ request binding;
- no accepted-requirement revision binding;
- no metric acceptance predicate;
- no validity-domain binding;
- no twin/design revision binding;
- no currentness proof;
- no solver-input/output identity check at the discharge transition;
- one converged request can fan out to unrelated simulation obligations.

**Migration:** ETK-3A removes discharge mutation from `evaluate_concept`. Simulation execution may still produce sensations/memory and candidate evidence, but never discharge.

### A2. Structural native-solver wrapper

`EngineeringManager::discharge_structural_check()` directly sets matching obligations to `Discharged` when the closed-form structural assessment passes, while attaching a free-form summary string.

**Migration:** retain `evaluate_structural()` as computation. Replace direct discharge with typed analytical/native-model candidate evidence and a later ETK admission policy.

### A3. Electrical native-solver wrapper

`EngineeringManager::discharge_electrical_check()` directly discharges a claim-matched obligation when the radial DistFlow assessment passes.

**Migration:** retain `evaluate_electrical()`; route any authority-bearing use through typed evidence admission.

### A4. Thermofluid native-solver wrapper

`EngineeringManager::discharge_thermofluid_check()` directly discharges a claim-matched obligation when the pipe-flow assessment passes.

**Migration:** retain `evaluate_thermofluid()`; route any authority-bearing use through typed evidence admission.

### A5. Shared native-discipline discharge helper

`EngineeringManager::discharge_obligation()` pushes a free-form evidence string and sets `ObligationStatus::Discharged` by claim equality. It is used by the following passing-assessment wrappers:

- `discharge_control_check()`;
- `discharge_circuit_check()`;
- `discharge_acoustic_check()`;
- `discharge_optical_check()`;
- `discharge_signal_check()`;
- `discharge_operations_check()`.

These are distinct analytical/model families with different validity envelopes. They must not share a generic string-based authority transition.

**Migration:** keep the existing `evaluate_*` methods as non-authoritative computation. Add an explicit native/analytical evidence contract that binds model identity, inputs, validity envelope, acceptance predicate, uncertainty, subject/twin/requirement/obligation revisions, and currentness before any discharge receipt can exist.

### A6. Formal-safety free-form discharge constructor

`ProofObligation::discharge(evidence_ref)` accepts an arbitrary string, appends it to `evidence_refs`, and sets the status to `Discharged`.

**Migration:** legacy compatibility may remain temporarily, but ETK-qualified engineering paths must never call it. A later formal-safety hardening should make discharge derived from typed receipts rather than a mutable field transition.

### A7. Public mutable obligation fields

`ProofObligation.status` and `ProofObligation.evidence_refs` are public. External callers can set `Discharged` or attach evidence strings directly without using `ProofObligation::discharge()`.

**Migration:** after downstream callers migrate, make authority-bearing state private or replace mutable discharge state with derived receipt applicability.

### A8. Example-level direct mutation

`examples/sovereign_design_loop.rs` directly assigns `ObligationStatus::Discharged` to an obligation.

**Migration:** examples must model the same authority boundary as production code. Demonstration code must not teach bypass patterns.

## B. Missing exact bindings

### B1. Requirement ↔ obligation

`EngineeringConcept::add_requirement()` creates a new `ProofObligation` from the requirement statement/evidence kind and then stores the requirement separately. The resulting obligation ID is not retained as an explicit binding to that requirement.

A later statement mutation can therefore make the semantic relationship ambiguous.

**Migration:** ETK-3B introduces an explicit immutable requirement/obligation link carrying content-addressed revisions.

### B2. Obligation ↔ simulation request

`EngineeringConcept.requirements`, `simulation_requests`, and `safety_case.obligations` are independent vectors. No first-class record says which exact request is intended to satisfy which exact obligation.

**Migration:** introduce an `EngineeringEvidencePlanV1` (name provisional) binding one accepted requirement revision, one obligation snapshot, one request ID, one evidence policy, one validity domain, and one twin/design revision.

### B3. Engineering concept ↔ twin/currentness

`EngineeringManager::evaluate_concept()` receives only `&mut EngineeringConcept`. Present twin revision/currentness is not part of the evaluation boundary, while `EngineeringReview` stores the optional twin separately.

**Migration:** authority-bearing admission/discharge must require an explicit context rather than inferring currentness from the concept.

## C. Authority amplification consumers

### C1. Safety-case closure

`SafetyCase::is_discharged()` returns true when every obligation's mutable status is `Discharged`.

Until this becomes receipt-derived, any bypass above can falsely close the safety case.

### C2. Deployment gating

`EngineeringReview::blocks_deployment()` uses `SafetyCase::is_discharged()` for blocking requirements. A false discharge can therefore affect deployment gating.

### C3. Placeholder Lean rendering

`EngineeringManager::formally_verify()` operates on already-`Discharged` obligations and creates a rendered Lean artifact from placeholder propositions/results. The source comments correctly state that this is not real formal verification, but an unjustified discharge can still cause an artifact to be emitted.

**Migration:** rename/reframe as rendering only, or require genuine formal-proof evidence before any formal-verification claim.

### C4. Swarm proof gossip

`EngineeringManager::broadcast_design_wisdom()` converts `Discharged` obligations into `SwarmProofMsg` records with `verified: true` and placeholder SMT/proof data.

This is an authority amplifier: local mutable status can become distributed proof-like metadata.

**Migration:** never set `verified: true` from mutable discharge status. Gossip should carry typed receipt/evidence identities and clearly separated verification state.

### C5. Technical-report language

`DocumentGenerator::generate_technical_report()` labels supplied proof artifacts as `Status: **DISCHARGED**` and says the design has been mathematically proven against structural invariants.

That wording can overstate placeholder/non-qualified proof artifacts.

**Migration:** report the actual authority level (`candidate`, `admitted`, `discharged`, `qualified`) and evidence class. Reserve “mathematically proven” for a checked formal-proof lineage.

## D. Separate model-authority bypasses

These do not directly set `Discharged`, but they can mutate models that later influence engineering decisions and therefore belong to the ETK migration program.

### D1. Metrology → live causal-model mutation

`process_metrology()` calls `calibrate_causal_model()` after an anomaly; the latter mutates causal conditional tables directly.

### D2. Dream consolidation → live causal-model mutation

`dream_consolidation()` directly adjusts causal safety priors and persists them.

**Migration:** field observations and dream/replay outputs must produce `ModelUpdateProposal` artifacts. Only validated/qualified updates create a new causal-model revision. Learning may be aggressive; engineering belief updates must remain conservative and auditable.

## E. ETK migration tranches

### ETK-3A — stop automatic external-simulation discharge

- remove `evaluate_concept()` mutation of `ObligationStatus`;
- add a regression theorem: converged simulation leaves safety obligations unchanged;
- keep simulation execution, HDC sensation, and episodic-memory behavior;
- make no claim that native-discipline bypasses are closed yet.

### ETK-3B — exact evidence plan

Add a first-class mapping:

```text
AcceptedRequirementRevision
        ↓
ProofObligationSnapshot
        ↓
SimulationRequest
        ↓
EvidencePolicy + ValidityDomain + TwinRevision
```

No vector-position or claim-string inference.

### ETK-3C — native analytical evidence

Define a separate admission class for Symthaea's native closed-form/analytical faculty solvers. Required bindings should include:

- solver/model identity and code revision;
- exact inputs and units;
- declared solver envelope / applicability domain;
- acceptance predicate;
- uncertainty/error model where applicable;
- exact subject/twin/requirement/obligation revisions;
- currentness;
- provenance/content digest.

A native analytical result must never be promoted to `ExternalSolver` merely to reuse the simulation path.

### ETK-3D — retire mutable discharge authority

After consumers migrate:

- stop using free-form `ProofObligation::discharge()` in engineering;
- make `SafetyCase` closure receipt-derived;
- remove/publicly seal direct mutable `status` authority;
- migrate deployment gating, proof rendering, swarm gossip, reports, and examples.

### ETK-4 — model-update authority

Replace direct metrology/dream model mutation with proposal → validation → qualified model revision transitions.

## Closure criterion

The bypass census is closed only when a repository-wide search demonstrates that no engineering path can create or amplify authoritative `Discharged`/`verified` state from raw solver/model outputs, free-form strings, claim matching, mutable public fields, or example shortcuts.

A clean search is necessary but not sufficient; exact-head tests and independent qualification remain separate requirements.
