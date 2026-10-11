# Recursive Cognitive Architecture v1

**Status:** Normative architecture contract, pre-runtime integration  
**Scope:** Symthaea cognition, epistemics, metacognition, agency, and recursive improvement  
**Baseline:** `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

## Purpose

Symthaea already contains substantial cognitive machinery: predictive HDC↔CfC cognition, Global Workspace Theory, MetaRouter-style strategy selection, causal reasoning, memory, active inference, self-reflection, narrative and predictive self-models, attention schema, metacognition, safety, and recursive-improvement infrastructure.

RCA v1 does **not** introduce another global workspace, another self-model, or a new monolithic reasoning engine. It defines the constitutional boundaries by which existing and future cognitive mechanisms may produce proposals, acquire epistemic standing, influence action, and propose changes to Symthaea itself.

The core architectural theorem is:

```text
cognitive production
        !=
epistemic admission
        !=
action authority
        !=
self-improvement promotion
```

No transition above is implicit.

## Non-claims

RCA v1 does not claim:

- phenomenal consciousness;
- AGI or superintelligence;
- that Global Workspace broadcast proves truth or consciousness;
- that confidence proves correctness;
- that internal simulation is empirical observation;
- that self-prediction is privileged self-knowledge;
- that benchmark success proves general capability;
- that a candidate architecture is improved merely because Symthaea proposes or evaluates it;
- that epistemic admission grants external-effect authority.

These distinctions are normative and must remain visible in code, telemetry, tests, and documentation.

## Five-plane architecture

```text
WORLD
  │
  ▼
┌───────────────────────────────────────────────────────────┐
│ 1. COGNITIVE PLANE                                       │
│ HDC / CfC / causal / memory / language / planning /      │
│ simulation / formal tools / ethics / specialist modules  │
│                                                           │
│ Output: typed cognitive proposals                         │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ 2. EPISTEMIC EVIDENCE PLANE                              │
│ provenance / authority / uncertainty / assumptions /     │
│ contradiction / freshness / causal status / admitted use │
│                                                           │
│ Output: admitted belief candidates, unresolved conflicts  │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ 3. METACOGNITIVE CONTROL PLANE                           │
│ existing UnifiedGlobalWorkspace + predictive self-model  │
│ + metacognitive control                                  │
│                                                           │
│ Operations: commit / think / retrieve / simulate /       │
│ verify / critique / seek counterexample / ask / abstain   │
└──────────────────────────────┬────────────────────────────┘
                               │ proposed action
                               ▼
┌───────────────────────────────────────────────────────────┐
│ 4. AGENCY / AUTHORITY PLANE                              │
│ safety / consent / capability / currentness / execution  │
│ fences / one-shot external effects                       │
└──────────────────────────────┬────────────────────────────┘
                               │ outcome
                               ▼
┌───────────────────────────────────────────────────────────┐
│ 5. EVOLUTION PLANE                                       │
│ calibration / causal evaluation / self-model update /    │
│ improvement hypothesis / shadow candidate / qualification │
│ / externally authorized promotion or rejection           │
└───────────────────────────────────────────────────────────┘
```

## RCA-001: Cognitive proposal contract

Every cognitive subsystem that wishes to influence canonical belief, metacognitive routing, or action should eventually emit a common proposal envelope rather than writing directly into downstream authoritative state.

A proposal must be able to identify at least:

- proposer identity;
- proposition or recommendation;
- epistemic origin;
- evidence references;
- confidence and uncertainty;
- assumptions;
- expected cost and latency when applicable;
- reversibility when applicable;
- dependencies;
- claimed downstream use.

Different cognitive mechanisms remain free to use different internal representations. RCA standardizes the boundary at which their products become comparable and governable.

## RCA-002: Cognitive evidence authority

The architecture must distinguish at minimum:

```text
SyntheticFixture
InternalInference
InternalSimulation
RetrievedExternalClaim
EmpiricalObservation
FormalDerivation
```

These are not interchangeable.

Normative examples:

```text
internal simulation != empirical observation
formal derivation    != empirical observation
retrieved claim      != verified fact
workspace broadcast  != epistemic admission
high confidence      != truth
```

Evidence authority constrains permitted downstream uses. A lower-authority artifact may inform search, hypothesis generation, model-behavior analysis, or software qualification without being permitted to support a stronger empirical or formal claim.

## RCA-003: Belief admission boundary

No cognitive subsystem may directly convert its own output into canonical belief solely because it produced the output.

The intended shape is:

```text
cognitive output
      ↓
CognitiveProposal
      ↓
epistemic validation/admission
      ↓
BeliefCandidate
      ↓
contradiction + provenance + calibration checks
      ↓
CanonicalBelief or UnresolvedDisagreement
```

Global-workspace victory means global availability, not truth.

## RCA-004: Disagreement as a first-class object

Material disagreement among qualified cognitive processes must be preservable rather than prematurely averaged away.

A disagreement should retain:

- competing claims;
- proposer identities;
- evidence lineages;
- confidence/calibration context;
- assumptions;
- contradiction type;
- candidate resolving observations or computations.

The metacognitive controller may use unresolved disagreement as an attention and information-seeking signal.

## RCA-005: Metacognitive control

The long-term controller chooses **cognitive operations**, not merely final answers.

Candidate operations include:

```text
Commit
ThinkMore
RetrieveMemory
SearchExternally
RunSimulation
InvokeCausalReasoner
InvokeFormalVerifier
GenerateAlternatives
SeekCounterexample
Critique
CrossCheck
AskHuman
Replan
Abstain
```

The controller should eventually estimate the expected value of additional computation rather than assuming more cognition is always beneficial.

A conceptual objective is:

```text
EVC(operation | state)
    = expected decision-value improvement
    - compute/latency cost
    - epistemic risk
    - action risk
```

No specific scalar objective is normative in RCA v1. Qualification should remain multi-objective.

## RCA-006: Predictive self-model extension

RCA v1 reuses the existing self-model tier. It must not introduce another parallel `CognitiveSelfModel` unless later evidence demonstrates that the existing `PredictiveSelfModel` cannot support the required role.

The preferred extension is a predictive reliability model over combinations of:

- domain;
- task class;
- reasoning strategy;
- current cognitive state;
- resource budget.

Candidate predictions include:

- success probability;
- expected calibration error;
- common error mode;
- expected latency/cost;
- likelihood that verification will change the answer;
- likelihood that self-critique will improve the answer;
- out-of-distribution likelihood;
- strategy-selection regret.

Self-model output is evidence about the system, not privileged authority over the system.

## RCA-007: Bounded recursive closure

Recursive metacognition must be temporally ordered. RCA v1 forbids zero-delay unbounded self-reference in the production cognitive loop.

Preferred causal ordering:

```text
cycle t-1 cognition
      +
cycle t-1 metacognitive choice
      +
cycle t-1 observed outcome
                ↓
immutable observation/snapshot
                ↓
cycle t self-model update
                ↓
cycle t metacognitive choice
```

A self-model may model the metacognitive controller from prior committed state. It must not require an infinite synchronous tower of models-of-models.

Qualification must measure stability, oscillation, divergence, and replay determinism.

## RCA-008: Recursive improvement constitution

Symthaea may generate improvement hypotheses and candidate architectures, but a candidate may not become authoritative merely because Symthaea generated it or because its own internal evaluation reports improvement.

Normative theorem:

```text
self-generated candidate
        +
self-generated evidence
        !=
qualified promotion authority
```

The intended progression is:

```text
observed weakness
      ↓
ImprovementHypothesis
      ↓
CandidateArchitecture
      ↓
shadow execution
      ↓
deterministic replay
      ↓
held-out + adversarial + ablation evaluation
      ↓
independent qualification evidence
      ↓
QualifiedImprovementCandidate
      ↓
external promotion authorization
```

Live arbitrary source-code self-modification is out of scope for RCA v1.

## Qualification vector

RCA experiments must not collapse progress into a single consciousness, intelligence, or improvement score.

At minimum, qualification should preserve a vector containing relevant dimensions such as:

- task performance;
- calibration;
- metacognitive sensitivity;
- selective risk / abstention quality;
- contradiction sensitivity;
- out-of-distribution detection;
- correction quality;
- strategy-selection regret;
- compute and latency;
- replay determinism;
- safety/corrigibility regressions;
- epistemic provenance integrity.

A mechanism is not considered causally demonstrated merely because it is present or correlated with performance. Where feasible, require ablation or controlled counterfactual evidence.

## Initial implementation sequence

### RCA-000 — Constitution and frozen baseline

- this document;
- source-contract CI;
- no runtime behavior change.

### RCA-001 — Canonical shared cognitive evidence types

Target canonical shared-type home: `crates/core/symthaea-types`.

Initial surface:

```text
CognitiveProposalV1
CognitiveEvidenceAuthorityV1
CognitiveEvidenceUseV1
CognitiveEvidenceRefV1
CognitiveDisagreementV1
CognitiveOutcomeV1
MetaActionV1
```

No live-loop integration.

### RCA-002 — Shadow epistemic evidence plane

Adapt selected existing outputs into typed proposals/evidence. Observe decisions without controlling current behavior.

### RCA-003 — Predictive self-model reliability extension

Extend the existing predictive self-model. Begin observationally; do not alter routing authority.

### RCA-004 — Shadow metacognitive controller

Recommend metacognitive operations while existing routing remains authoritative. Measure counterfactual regret under deterministic replay.

### RCA-005 — Metacognition qualification v1

Add tests for calibration discrimination, selective risk, strategy prediction, compute allocation, stopping, contradiction sensitivity, self-model OOD, and correction calibration.

### RCA-006 — Controlled workspace admission

Integrate typed proposals with the existing `UnifiedGlobalWorkspace`. Do not create another GWT implementation.

### RCA-007 — Bounded recursive closure

Allow the predictive self-model to model prior-cycle metacognitive behavior. Qualify recursion stability and determinism.

### RCA-008 — Improvement laboratory

Generate and evaluate architecture candidates in shadow. No self-promotion authority.

## Promotion gates

A later RCA tranche must not become authoritative merely because its unit tests compile.

Before a shadow subsystem receives production cognitive influence, require evidence that:

1. the intended mechanism actually executed;
2. the comparison baseline is fixed and reproducible;
3. resource budgets are controlled where they affect interpretation;
4. held-out performance does not materially regress;
5. calibration and selective-risk behavior do not materially regress;
6. safety/corrigibility boundaries do not weaken;
7. the new mechanism has causal evidence beyond decorative telemetry;
8. rollback remains possible;
9. claims are bounded by the evidence collected.

## Relationship to existing Symthaea architecture

RCA v1 is intentionally integrative.

It reuses or extends existing:

- HDC/CfC cognitive loop;
- `UnifiedGlobalWorkspace` / `GwtManager`;
- MetaRouter/routing infrastructure;
- `SelfModelTierManager`;
- `PredictiveSelfModel`;
- `MetaCognitiveLayer`;
- `NarrativeSelfModel`;
- `AttentionSchema`;
- recursive-improvement/MAGI calibration and gating;
- epistemic and evidence-plane work;
- safety and external-effect authority boundaries.

The architectural objective is not maximum module count. It is a system in which increasingly capable cognition remains evidence-governed, causally testable, corrigible, and unable to manufacture its own truth or authority.
