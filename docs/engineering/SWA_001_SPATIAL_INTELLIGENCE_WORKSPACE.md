# SWA-001 Spatial Intelligence & Adaptive Workspace Architecture

Status: proposed integration architecture

This document records where workspace/environment intelligence belongs in the existing Symthaea/Mycelix stack. The goal is composition, not another parallel domain engine.

## Existing related work — important discovery

This capability is more developed than the initial proposal suggested.

### Symthaea substrate

The repository already contains:

- `symthaea-fabrication-kernel::building` for thermal load, structural stress, occupancy, comfort, and energy consumption across multi-scale horizons.
- `symthaea-digital-twin` for auditable engineered-system twin state and telemetry primitives.
- `symthaea-engineering` as a composition facade combining digital twins, simulation, causal reasoning, formal safety, materials aging, memory, and the Global Workspace.
- `symthaea-sim-bridge` for solver-agnostic FEA/CFD/multibody/circuit/process simulation.
- spatial cognition and perception infrastructure.
- resource-allocation and active-inference machinery.
- institutional failure learning lineage for prediction -> outcome -> error -> revised constraint.

The existing `TwinState` is already especially close to the required primitive: it stores identity, asset class, telemetry, prediction residuals, epistemic/aleatoric uncertainty, free-energy breakdown, health, and intervention candidates.

### Mycelix substrate

Mycelix already has an unusually direct environment/commons substrate:

- `mycelix-workspace`: ecosystem orchestration, shared bridge infrastructure, SDKs, routing, migration, and hApp composition.
- `mycelix-commons`: property, housing, maintenance, cooperative membership, community land trusts, governance, water, food, transport, mutual aid.
- `mycelix-energy`: energy projects, investment/ownership transitions, grid production/consumption and peer-to-peer energy coordination.
- `mycelix-hearth`: household/kinship coordination, shared resources, care, autonomy, rhythms and presence.
- `mycelix-identity`: identity, recovery and assurance infrastructure.
- `mycelix-attribution`: privacy-preserving usage receipts, attestations and stewardship lineage.
- `mycelix-governance` and shared bridge infrastructure for authorized decisions and cross-cluster routing.

Therefore SWA should **not** create new housing, energy, identity, commons, or governance stacks. It should connect these existing systems through a narrow evidence/decision boundary.

## Composition

A workspace is modeled as a coupled socio-technical environment:

PhysicalState
-> WorkspaceRepresentation
-> Predictions
-> CandidateInterventions
-> CounterfactualEvaluation
-> Human/authorized Decision
-> Intervention
-> Outcome
-> PredictionError
-> IFL ConstraintCandidate

### PhysicalState

Geometry and zones, thermal conditions, acoustic conditions, lighting, electrical load/power quality, structural condition, water/resource state, network dependencies, and equipment health.

### UseState

Aggregate occupancy, zone utilization, room/resource availability, demand patterns, and scheduling constraints. Individual performance inference is explicitly out of scope.

### HumanAgencyState

Explicitly supplied preferences, accessibility requirements, privacy choices, collaboration/deep-work preferences, and sensing consent. These are agency inputs, not latent personality labels.

### EconomicState

Lifecycle cost, utilization, maintenance burden, fixed commitments, stranded-cost exposure, reconfiguration cost, exit cost, and alternative-use value. Long-lived commitments must expose uncertainty rather than assuming demand.

### ResilienceState

Dependencies, spare capacity, degraded modes, islanding/fallback options, recovery time, and concentration risk.

### LearningState

Prediction, actual outcome, prediction error, mechanism hypotheses, counterevidence, counterfactuals, and learned constraints with complete lineage.

## Candidate types

The first adapter layer can introduce:

- WorkspaceAsset
- WorkspaceZone
- WorkspaceObservation
- WorkspacePreference
- WorkspaceIntervention
- WorkspacePrediction
- WorkspaceOutcome
- WorkspacePredictionError
- WorkspaceConstraintCandidate

These should reference existing engineering, digital-twin, temporal-prediction, and IFL types where practical.

## Boundary architecture

**Symthaea owns inference and simulation:**

observe -> represent -> predict -> simulate -> compare -> propose -> learn

**Mycelix owns durable coordination context:**

identity -> consent -> authorization context -> decision -> receipt -> contestability -> durable provenance

A Mycelix receipt must never turn a Symthaea prediction into an observation or grant physical authority.


## Provenance gate — SWA-004

The reference fixture now treats provenance as part of the value, not metadata that can be inferred later.

- ScenarioAssumption means an authored fixture/design assumption.
- ModelDerivedPrediction is reserved for an intervention-specific model output.
- ObservedOutcome is reserved for measured post-intervention state.
- DerivedResidual is computed from typed prediction/outcome pairs.
- UnresolvedEvidence remains contradictory without forced resolution.
- UnknownUnmodeled is explicit when no model supports a dimension.

This distinction follows the same basic discipline as the W3C PROV model: provenance describes entities, activities, agents, and derivations rather than treating all values as interchangeable facts. See the W3C PROV model and ontology. 

SWA-003 currently demonstrates an important negative capability: its intervention dimension values are scenario assumptions, while BuildingTwin outputs remain building-model evidence. The fixture therefore refuses to call those intervention dimensions a counterfactual until an intervention-specific model actually produces them.

That is intentional. Unknown is safer than fabricated precision.

## Multi-objective discipline

Do not reduce the workspace to a single opaque optimization score.

Keep separate dimensions such as:

- human agency
- privacy
- accessibility
- comfort
- deep-work capacity
- collaboration
- adaptability
- resilience
- lifecycle cost
- resource efficiency
- environmental impact
- maintenance burden

Candidate interventions should expose tradeoffs and uncertainty so humans/authorized governance can choose among them.

## WeWork-derived reversibility invariant

For any proposed long-lived workspace commitment, preserve:

1. expected demand range
2. commitment duration
3. fixed vs variable cost
4. utilization assumptions
5. maintenance burden
6. reconfiguration cost
7. exit/termination cost
8. alternative-use value
9. downside/liquidity scenario
10. evidence supporting the assumptions

The system should be able to ask whether a lower-commitment or more modular design achieves similar outcomes before recommending an irreversible commitment.

## Privacy and agency invariants

- No covert sensing.
- No employee productivity scoring.
- No inferred sensitive traits from ambient environmental data.
- Preferences are not blanket consent.
- Aggregate environmental telemetry must remain distinct from individual records.
- Recommendations never become authorization automatically.
- Physical actuation requires an explicit authorized control boundary.

## Qualification path

G0: deterministic schemas and serialization
G1: replayable predictions
G2: exact prediction/outcome pairing
G3: uncertainty and contradiction preservation
G4: deterministic counterfactual fixtures
G5: privacy/agency invariant tests
G6: longitudinal recurrence and reconfiguration-cost fixtures
G7: Mycelix receipt projection preserving complete lineage

## Implementation rule

Prefer adapters around existing crates over a new physics or simulation engine. The workspace layer should become a composition boundary connecting physical models, cognitive inference, simulation, economic/resilience reasoning, and Mycelix durable coordination.

## Immediate next implementation slice

Build one deterministic fixture spanning a small building:

1. a `BuildingTwin` supplies physical predictions and residuals;
2. `symthaea-digital-twin::TwinState` supplies uncertainty/health/intervention context;
3. an intervention candidate changes a zone-level configuration;
4. a counterfactual evaluator compares at least comfort, energy, resilience, cost and reversibility independently;
5. IFL records prediction -> outcome -> error -> constraint lineage;
6. Mycelix records consent/authorization context and a durable decision receipt;
7. replay proves that the same fixture produces the same decision evidence without requiring the same authority holder.

This fixture should be the reference integration test before any real-world actuation work.

Related: Symthaea issue #6624; Mycelix issue #3634.

## Calibration and model-history gate — SWA-005

Calibration must produce a **new parameter/model revision**, never mutate the historical prediction that exposed the residual.

The minimum lineage is:

prediction(v1) -> observed/synthetic outcome -> residual(v1) -> calibration activity -> parameter set(v2) -> prediction(v2)

The calibration activity must identify:

- source model revision;
- source residual;
- previous parameter set;
- new parameter set;
- bounded parameter adjustment;
- calibration activity identity.

A revised model may improve future predictions, but it cannot rewrite the historical prediction, outcome, or residual.

This also keeps model credibility concerns separate: verification asks whether the implementation behaves according to its mathematical specification; validation asks whether it represents the intended real-world system; uncertainty quantification describes sensitivity and uncertainty in inputs and outputs. ASME's VVUQ framework explicitly treats these as distinct concerns. See https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty.

For digital twins, remaining discrepancy should not automatically be treated as parameter error. Model-form uncertainty can arise from abstraction and omitted physical processes, so a future calibration layer should be able to preserve residuals as unresolved/model-form discrepancy rather than forcing every residual into parameter updates. Recent digital-twin calibration research explicitly distinguishes model-form uncertainty from parameter uncertainty. See https://arxiv.org/abs/2609.10171.

The current SWA-005 fixture is intentionally a deterministic laboratory fixture, not a claim of physical validation.

## Validation freshness and revalidation frontier — SWA-006

Validation evidence is immutable historical evidence. A later model or dependency revision must not rewrite the record that was produced under the earlier revision.

SWA-006 therefore separates:

1. **Historical validation evidence** — the exact evidence produced at a point in model history.
2. **Dependency fingerprint** — model, parameter set, solver, scenario, dataset, uncertainty model, and context-of-use revisions used by that evidence.
3. **Evidence freshness** — a derived view of whether that historical evidence remains applicable to the current dependency frontier.
4. **Revalidation requirement** — the minimal reason and dependency change that must be addressed before relying on the evidence for the current revision.

The important invariant is:

**evidence can become stale without becoming historically false.**

For example:

- changing the model revision makes model-dependent evidence stale;
- changing only the parameter set identifies a parameter-lineage change rather than silently classifying the residual as model-form error;
- changing the uncertainty model invalidates the uncertainty characterization independently;
- changing the context of use requires applicability reassessment;
- changing the solver can require fresh verification/validation without mutating historical results;
- incomplete provenance yields `Unknown`, never an implicit pass.

The resulting flow is:

`historical evidence -> dependency fingerprint -> current dependency comparison -> freshness -> revalidation frontier`

The revalidation frontier is intentionally analogous to a build dependency graph: an upstream change propagates only to evidence that actually depends on that revision. This should eventually allow Sol Atlas to answer **which evidence must be regenerated, why, and which evidence remains applicable** without rewriting historical provenance.

This is compatible with W3C PROV's distinction between entities, activities, agents, revisions, invalidation, and derivation. PROV explicitly treats revisions as new entities and supports provenance chains rather than mutable histories. See the W3C PROV Model Primer and PROV ontology.

## Dependency closure — SWA-007

SWA-006 exposed a subtle failure mode: reporting only the first changed dependency loses information when several upstream entities change simultaneously.

SWA-007 therefore makes the **complete dependency delta** explicit and associates each evidence dimension with its own dependency closure.

A change is propagated only when it intersects the evidence's declared closure.

Examples:

- a solver revision can stale comfort/energy evidence that depends on solver output;
- the same solver revision need not stale an independently established resilience evidence record that does not depend on the solver;
- a context-of-use revision affects applicability evidence;
- a parameter revision affects evidence that actually depends on the parameter set;
- incomplete provenance produces `Unknown`, never `Valid`.

This gives the revalidation frontier a more precise meaning:

`current dependency delta ∩ evidence dependency closure = affected evidence`

The system consequently preserves both **completeness** and **minimality**: all changed dependencies remain visible, while unrelated changes do not create unnecessary revalidation work.

This is also closer to the provenance semantics of W3C PROV, where derivation expresses how a generated entity depends on prior entities and revision/invalidation are explicit provenance relations rather than implicit mutable state.


## Claim-level provenance — SWA-008

SWA-007 establishes dependency closure for validation evidence. SWA-008 moves the same discipline one semantic layer upward: a system should not merely know that evidence exists; it should know **which exact statement that evidence supports**.

An EvidenceClaim therefore records:

- a stable claim identity;
- the kind of claim being made;
- explicit links to supporting, contradicting, or qualifying evidence;
- the dependency fingerprint from which the claim was derived;
- an explicit non-authorization boundary.

The relation types are intentionally distinct:

- **Supports** means the evidence provides positive support for the claim;
- **Contradicts** means the evidence provides counterevidence;
- **Qualifies** means the evidence narrows the conditions under which the claim applies, without asserting that the underlying claim is false.

This prevents a common provenance failure where a boundary condition is incorrectly treated as either proof or refutation.

Claim status is derived rather than stored as mutable truth:

claim provenance -> current dependency comparison -> claim status

The fixture distinguishes:

- Valid — supporting evidence exists and the provenance remains current;
- Stale — the claim's dependency frontier changed;
- Unknown — provenance identity is incomplete or cannot be reconciled;
- Unsupported — no supporting evidence is linked;
- Contested — support and contradiction coexist.

Critically, Contested does not select a winner and Qualifies does not erase support. The graph preserves the epistemic state so a later decision process can inspect the underlying evidence.

This gives Sol Atlas a more useful primitive than a generic confidence score: it can eventually answer **what exactly are we claiming, what evidence supports it, what evidence challenges it, under which dependency revisions, and what changed since the claim was established?**

This follows the same provenance direction as W3C PROV, which models entities, activities, derivations, revisions, and invalidation explicitly rather than relying on mutable metadata. ASME's current VVUQ portfolio likewise treats verification, validation, uncertainty quantification, and model lifecycle as distinct credibility concerns rather than collapsing them into a single result.

The next natural seam is **provenance slicing**: given a downstream claim, compute the minimal reproducible chain of evidence and dependencies required to reproduce or challenge that claim.


SWA-008 also makes claim freshness closure-specific rather than using the entire validation fingerprint. A solver change therefore stales a prediction-validation claim that depends on solver output, while leaving an applicability claim untouched when its declared closure does not include the solver. This preserves the same completeness/minimality principle established by SWA-007 at the claim layer.


## Provenance slicing — SWA-009

SWA-008 made claims explicit. SWA-009 makes their justification queryable.

Given a downstream claim, the fixture computes the reachable provenance subgraph containing only the entities and derivations required to reproduce or challenge that claim. Unrelated provenance remains outside the slice.

The reference chain is:

claim -> validation evidence -> prediction -> model/parameters/scenario

with validation also linked to:

validation -> dataset/context-of-use

The resulting slice exposes a dependency frontier that can be inspected independently of the rest of the system.

This creates two important operations for Sol Atlas:

1. **Reproduce** — start from the claim and recover the minimum upstream evidence/model/context chain needed to recreate the result.
2. **Challenge** — inspect the same slice to identify which upstream entity, dependency revision, qualification, or counterevidence could weaken or invalidate the claim.

The slice preserves ContradictedBy and QualifiedBy edges rather than resolving them. That is deliberate: provenance answers what led to a claim; it does not silently decide the dispute.

This is closely aligned with W3C PROV's graph-oriented model of entities, activities, derivations, collections, and provenance bundles, including the ability to represent provenance across linked bundles.

The architectural progression is now:

validation evidence -> dependency closure -> evidence claim -> provenance slice

The next useful constraint is **slice completeness**: the system should be able to prove that a purported minimal slice contains every dependency required by the claim's declared closure, and reject a slice that omits a required upstream entity.


## Provenance-slice completeness — SWA-010

SWA-009 establishes reachability. SWA-010 adds a second invariant: **reachability is not sufficient evidence of reproducibility**.

A provenance slice must pass two tests:

- **Completeness:** every dependency kind required by the claim is present.
- **Minimality:** the slice contains no dependency kind outside the claim's declared closure.

The fixture emits a deterministic completeness certificate with explicit missing and unrelated dependency sets.

This gives the provenance layer a fail-closed property:

- missing model/parameter/scenario/dataset/context provenance is a failed completeness check;
- unrelated provenance cannot silently inflate the justification for a claim;
- the certificate itself remains evidence about the slice, not authorization.

The resulting architecture is now a small epistemic build system:

claim -> provenance slice -> completeness certificate -> dependency frontier

A future implementation can replace the fixture's simple node-kind closure with the actual typed dependency graph while preserving the same contract.


## Reproducibility witnesses — SWA-011

SWA-010 proves that a provenance slice is structurally complete. SWA-011 makes the next step executable: a complete slice can carry a deterministic replay witness.

A **ReproducibilityWitness** binds five things without granting any of them authority:

- claim identity;
- execution recipe and algorithm revision;
- canonicalized dependency revisions;
- canonical input values;
- the deterministically derived artifact.

The fixture canonicalizes dependency and input ordering before deriving fingerprints. Identical manifests therefore produce identical witness identities, while a dependency revision change produces a different witness. Missing dependencies or inputs fail closed rather than producing a partial witness.

The fixture currently uses a small dependency-free FNV-1a fingerprint because the example is intentionally self-contained. This is explicitly a fixture fingerprint, not a cryptographic commitment. A production interchange layer should substitute a cryptographic digest without changing the canonical-manifest contract.

The important architectural distinction is:

claim -> complete provenance slice -> canonical replay manifest -> deterministic execution -> reproducibility witness

This also gives Sol Atlas a concrete answer to **“can this claim actually be reproduced from the evidence we say supports it?”** rather than treating provenance as descriptive metadata alone. W3C PROV explicitly includes reproducibility and versioning among the provenance requirements it supports, and its bundle model permits provenance to be independently established and linked across provenance boundaries. ASME's VVUQ lifecycle guidance likewise places verification, validation, and uncertainty work inside the model lifecycle rather than treating a validation result as permanently detached from model evolution.

The next seam is **counterevidence witnesses**: supporting, qualifying, and contradicting claims should each be independently replayable from their own complete provenance slices, allowing the system to preserve disagreement without collapsing it into a single confidence value.
