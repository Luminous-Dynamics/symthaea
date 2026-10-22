# SPINE-000 — Cognitive Spine Causal Census

**Status:** architecture/evidence census; no production cognition rewiring

**Baseline:** `4bad8af72ff775e7c869b6df83faba718a339a36`

**Purpose:** establish what is actually connected, what is causally load-bearing, and which existing seam should become the Symthaea Cognitive Spine before adding or promoting more cognitive machinery.

## 1. Constitutional rule

SPINE-000 does **not** infer functional importance from code presence, module names, theory labels, or architectural intent.

A component earns promotion into the Cognitive Spine only through a chain of evidence:

1. the component is reachable from the live cognitive cycle;
2. its inputs and outputs are attributable;
3. its output can affect a downstream state or decision;
4. disabling, perturbing, or substituting the component changes a preregistered observable in the predicted direction;
5. the effect survives an appropriate control/null comparison.

The allowed census classes are:

- `LOAD_BEARING` — measured causal effect on a preregistered observable;
- `CONNECTED_UNQUALIFIED` — live/reachable with a downstream path, but no qualifying ablation yet;
- `CONNECTED_NULL` — live/reachable, but the current ablation/evidence lineage found no meaningful causal effect;
- `DORMANT` — implemented but not on the current live execution path;
- `RESEARCH_ONLY` — intentionally experimental/telemetric and not authoritative production cognition;
- `DUPLICATE_OR_OVERLAPPING` — overlapping ownership or parallel implementation that prevents clean causal attribution.

`NOT_DEMONSTRATED` is not equivalent to `DOES_NOT_WORK`.

## 2. Current live spine candidate

The current `CognitiveLoopService` documentation already describes a partial spine:

```text
input
  -> HDC encode
  -> CfC temporal processing
  -> multi-horizon prediction
  -> prediction error
  -> CfC learning + HDC attention update
  -> semantic/episodic memory + causal enhancement
  -> active inference / reasoning / Broca
  -> output
```

This path should be treated as the incumbent integration target, not replaced by a new parallel `UnifiedMind` abstraction.

The current dynamics phase is already ordered around:

```text
OBSERVE
  -> COMPUTE subsystem proposals
  -> memory binding
  -> CfC planning / world model
  -> active inference
  -> reasoning
  -> metacognition
  -> training / Broca / post-processing
```

SPINE-001 should narrow and type this path rather than add a second orchestrator.

## 3. Existing proposal-based seam

The repository already contains a `CognitiveSubsystem` interface and `CycleSnapshot` / `SubsystemOutput` protocol.

Its intended model is:

```text
A. OBSERVE
   immutable CycleSnapshot

B. COMPUTE
   CognitiveSubsystem::process(snapshot)
      -> SubsystemOutput

C. INTEGRATE
   OutputCollector consensus
      -> one integrated proposal

D. DECAY / HOMEOSTASIS
```

This is the strongest existing seam for SPINE-001 because it provides:

- immutable observation;
- proposal rather than direct mutation;
- subsystem attribution;
- a common integration surface;
- future WASM/hot-swap compatibility;
- a natural place to record causal-load telemetry.

However, the migration is incomplete. Several managers explicitly document that they run alongside existing inline code in a **dual-write bridge**. `CognitiveLoopService` itself also labels the collector path as dual-write alongside direct mutation.

Therefore the proposal path is currently `CONNECTED_UNQUALIFIED`, not yet a unique causal authority.

## 4. Current manager layer

`src/cognitive_loop/managers/` describes managers that group 75+ `CognitiveLoopService` subsystems and implement `CognitiveSubsystem`.

Code search confirms implementations for, among others:

- perception;
- vision;
- memory;
- learning;
- language;
- reasoning;
- drives;
- survival;
- trust;
- social fabric;
- time;
- thermodynamic state;
- multimodal processing;
- spectral processing;
- hypervisor/supervision;
- additional domain managers.

This is an important positive result: SPINE-001 does **not** need to invent a generic cognitive-module trait. A usable protocol already exists.

The research task is instead to determine which manager outputs are actually consumed, duplicated by inline code, overwritten later, or causally null.

## 5. Evidence inherited from the July adversarial ablation lineage

The July 2026 adversarial re-grade is the strongest existing causal-load evidence found in the repository.

It reports:

- 13/15 tested loop subsystems with approximately null causal load on the measured outputs;
- `meta_cognition` as load-bearing for the measured consciousness-level output;
- `embodied_cognition` as load-bearing for the measured Psi output;
- GWT, prefrontal, predictive processing, phi-attention, phenomenal binding and dream replay as null in that evidence lineage;
- prediction error frozen at `1.0000` across measured arms;
- tick-rate changes from 50 Hz to 1 Hz producing no output change to four decimal places;
- divergence between facade and loop cognition state.

These results constrain SPINE-001:

1. Do not make GWT the spine merely because it is architecturally attractive.
2. Do not treat configured CfC/predictive machinery as load-bearing until prediction error is demonstrably dynamic.
3. Preserve and strengthen metacognitive integration because it has existing measured causal load.
4. Require all promoted spine components to acquire an ablation witness.

## 6. HDC status and role

HDC is a strong candidate for the shared **associative substrate**, not for semantic authority.

Existing code provides:

- canonical high-dimensional representations;
- bind / bundle / similarity / permutation operations;
- semantic and temporal encoders;
- episodic and semantic memory integration;
- Hebbian/adaptive learning signals;
- bridges into active inference and coalition formation;
- compressed/full-vector memory support.

SPINE rule:

```text
exact typed admissibility
    -> HDC candidate search / association
    -> exact typed validation
```

HDC similarity MUST NOT itself establish truth, authority, world promotion, causal validity, or permission to assert.

This generalizes the existing typed-routing lesson: approximate similarity is useful **inside** an exact admissible namespace.

## 7. CfC / HDC-LTC status and role

The current loop documents CfC as its temporal predictor. The repository also contains `HdcLtcBridge`, which can act as a drop-in temporal backend and supports:

- configurable HDC dimensionality;
- fast lower-dimensional configurations;
- 16,384D full configurations;
- optional adaptive dimensionality scaling;
- online adaptation;
- state snapshot/restore for evaluative prediction paths.

The HDC-LTC unified network is also reused in multiple physical-control domains, which supports treating it as a reusable temporal primitive.

SPINE rule:

```text
CfC / HDC-LTC predicts state evolution.
It does not decide what is true.
```

A temporal prediction must remain typed as a prediction until observation/evidence and authority logic permit promotion.

## 8. Provisional census

The following is intentionally conservative. `CONNECTED_UNQUALIFIED` means the code path exists but SPINE-000 has not yet established a qualifying causal witness.

| Component / family | Provisional class | Reason |
|---|---|---|
| Core HDC encoding | CONNECTED_UNQUALIFIED | live loop input substrate; functional contribution needs isolated comparison |
| CfC temporal network | CONNECTED_UNQUALIFIED | live documented path; earlier PE/tick-rate evidence prevents load-bearing claim |
| HDC-LTC bridge | CONNECTED_UNQUALIFIED | implemented alternative backend; requires controlled substitution evidence |
| Meta-cognition | LOAD_BEARING (bounded) | July ablation changed measured consciousness-level output |
| Embodied cognition | LOAD_BEARING (bounded) | July ablation changed measured Psi output |
| GWT | CONNECTED_NULL (July lineage) | disabling produced approximately null measured deltas |
| Prefrontal path | CONNECTED_NULL (July lineage) | disabling produced approximately null measured deltas |
| Predictive-processing feature | CONNECTED_NULL (July lineage) | disabling null and PE frozen in the cited lineage |
| Phi-attention | CONNECTED_NULL (July lineage) | disabling produced approximately null measured deltas |
| Phenomenal binding | CONNECTED_NULL (July lineage) | disabling produced approximately null measured deltas |
| Dream replay | CONNECTED_NULL (July lineage) | disabling produced approximately null measured deltas |
| Proposal/manager protocol | CONNECTED_UNQUALIFIED | real collector/integration path, but dual-write with inline mutations |
| Direct inline manager-equivalent paths | DUPLICATE_OR_OVERLAPPING | coexist with proposal managers during migration |
| Structural Phi / consciousness research metrics | RESEARCH_ONLY for spine authority | useful research telemetry; not validated as authority/control signal |
| Broca | CONNECTED_UNQUALIFIED for full spine | live language path; separate Broca/XLI qualification program applies |
| Broca authority kernel | candidate authority boundary | must remain independent from HDC/CfC confidence |

The table is not a permanent verdict. Every entry is intended to be mechanically upgradable/downgradable by evidence.

## 9. SPINE-000 executable census schema

The next implementation tranche should emit one record per subsystem per evidence run:

```text
CausalCensusRecord
  subsystem_id
  implementation_path
  feature_gate
  schedule_interval
  reads[]
  proposals[]
  direct_writes[]
  downstream_consumers[]
  duplicate_paths[]
  enabled_in_arm
  ablation_kind
  measured_observables[]
  expected_effect
  observed_effect
  effect_size
  status
  evidence_lineage
```

No component may receive `LOAD_BEARING` solely because `process()` executed or because it emitted a nonzero proposal.

## 10. Required SPINE-000 measurements

The executable census should answer four different questions and keep them separate:

### Reachability

Did the component execute on the tested path?

### Proposal influence

Did its proposal survive integration and change integrated state?

### Downstream influence

Did that changed state reach a later cognitive decision/output rather than being overwritten or ignored?

### Causal load

Did disabling/substituting the component change a preregistered observable relative to its matched control?

This distinction prevents `called == useful` and `nonzero output == causally important` mistakes.

## 11. SPINE-001 target boundary

SPINE-001 should use the existing snapshot/proposal architecture, not create a new universal mind.

The target is:

```text
TypedCognitiveFrame
       |
       +--> HDC associative projection
       +--> temporal state
       +--> evidence/world/authority references
       |
       v
CognitiveSubsystem specialists
       |
       v
attributed CognitiveProposal set
       |
       v
proposal arbitration
       |
       v
one typed state transition
```

### TypedCognitiveFrame

The authoritative frame should contain typed semantic/world/evidence state and may contain HDC/temporal projections as non-authoritative accelerators.

### CognitiveProposal

Proposals should become semantically typed rather than collapsing immediately into generic scalar deltas. Candidate families include:

- memory recall;
- temporal prediction;
- causal hypothesis;
- attention request;
- action candidate;
- epistemic update;
- learning update;
- realization request.

### Arbitration

Consensus averaging may remain appropriate for scalar homeostatic modulation, but it must not be the universal integration rule for semantic/epistemic proposals.

For example, two contradictory causal claims cannot safely be averaged into a third claim.

SPINE-001 therefore needs typed arbitration policies by proposal kind.

## 12. CPU-native requirement

The reference spine must remain CPU-native.

Performance work should exploit existing properties rather than require a large neural model:

- HDC SIMD/binary operations;
- adaptive HDC dimensionality;
- small temporal experts rather than one monolith;
- co-prime/sparse subsystem scheduling where validated;
- HDC routing before expensive specialist execution;
- cached memory/retrieval projections;
- uncertainty-triggered compute escalation.

Candidate policy:

```text
low uncertainty
  -> small HDC dimension
  -> minimal specialists
  -> cheap deterministic realization

high uncertainty / novelty
  -> increase HDC dimension
  -> activate retrieval / causal / temporal specialists
  -> optional neural specialist
  -> stronger verification
```

The policy itself must be benchmarked rather than assumed efficient.

## 13. Promotion gates for later spine tranches

A proposed component integration should not advance merely because aggregate benchmark score improved.

Minimum evidence package:

1. exact subject/code identity;
2. matched enabled/disabled or candidate/control arms;
3. raw intermediate telemetry;
4. preregistered target observable;
5. no regression in authority-integrity invariants;
6. CPU latency/RAM accounting;
7. independent postflight/lineage check;
8. negative result preserved if the component is null.

For approximate modules (HDC similarity, CfC prediction), also require calibration/error distributions and explicit failure states.

## 14. Immediate patch sequence

### SPINE-000A — executable registry

Add a read-only registry that enumerates the live manager/subsystem set, schedule interval, feature gate, and proposal flags. It must not change cognition.

### SPINE-000B — proposal influence receipts

For each manager execution, record whether:

```text
executed
-> emitted proposal
-> proposal admitted
-> integrated value changed
-> downstream value consumed
```

### SPINE-000C — duplicate-path census

Map proposal-manager responsibilities to surviving direct inline mutation paths. This is required before deleting either path.

### SPINE-000D — matched ablation harness

Generalize the existing subsystem-ablation strategy so every spine candidate can be run under matched enabled/disabled arms with identical inputs/seeds.

### SPINE-000E — baseline seal

Freeze the resulting census and classify every candidate as one of the six census states.

Only then start behavioral SPINE-001 migration.

## 15. First SPINE-001 migration candidate

The first migration should be intentionally small and measurable.

Recommended candidate: **metacognition proposal path**, because the historical ablation already gives it a bounded causal witness.

Goal:

```text
existing direct metacognitive modulation
vs
proposal-only metacognitive modulation
```

Requirements:

- byte/semantic-equivalent inputs;
- exact attribution of the proposal;
- direct-write path disabled only in the candidate arm;
- same downstream observables;
- no authority changes;
- no global rewrite of CognitiveLoopService;
- restore/fallback path preserved until qualification.

If proposal-only metacognition reproduces the intended effect, the Cognitive Spine gains its first causally qualified migrated specialist.

## 16. Non-goals

SPINE-000 does not:

- claim consciousness;
- promote structural Phi to authority;
- merge `ContinuousMind` into another facade;
- delete existing inline paths;
- make HDC similarity authoritative;
- make CfC predictions factual;
- change Broca/XLI qualification criteria;
- enable every dormant subsystem;
- interpret code existence as intelligence.

## 17. Success condition

SPINE-000 succeeds when Symthaea can answer, from machine-readable evidence:

> Which cognitive components ran, what did they propose, what state did they actually change, what downstream decisions consumed that change, and which effects disappear under matched ablation?

That evidence—not module count—becomes the foundation for the Cognitive Spine.
