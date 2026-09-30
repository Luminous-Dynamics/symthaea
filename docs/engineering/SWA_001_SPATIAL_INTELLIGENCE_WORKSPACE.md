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
Related: Symthaea issue #6624; Mycelix issue #3634.## Calibration and model-history gate — SWA-005

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


## Counterevidence witnesses — SWA-012

SWA-011 made a single claim replayable. SWA-012 makes disagreement replayable as well.

A `CounterevidenceRecord` links independently identified reproducibility witnesses to a claim using three explicit relations:

- **Supports** — the witness supplies evidence in favor of the claim;
- **Qualifies** — the witness narrows or conditions the claim without asserting that it is false;
- **Contradicts** — the witness supplies evidence against the claim.

The fixture preserves every witness identity and deliberately does not resolve conflicts into a score or winner. A contradiction therefore yields `Contested`, while support plus qualification yields `Qualified`. Qualification alone does not become support, and absence of evidence remains `Unsupported`.

This is a useful distinction for Sol Atlas because a claim can now carry multiple independently replayable epistemic paths:

claim -> support witness
claim -> qualification witness
claim -> contradiction witness

Each witness can point back to its own complete provenance slice and dependency frontier. A later model revision can therefore invalidate one branch without silently deleting the historical fact that another branch contradicted it.

This is consistent with the W3C provenance model's emphasis on validity and consistency checking: PROV defines constraints over provenance histories and provides normalization/equivalence machinery so provenance can be checked rather than merely stored. The provenance overview also explicitly lists reproducibility, versioning, procedures, and derivation as core provenance requirements. citeturn0search1turn0search5

The architectural progression is now:

claim -> provenance slice -> completeness certificate -> replay witness -> support/qualification/contradiction set

The next seam is **provenance integrity validation**: rather than merely constructing these structures, the fixture should reject impossible histories such as duplicate witness identities, missing upstream nodes, contradictory dependency identities, or derivation cycles. That moves the system from provenance storage toward provenance verification.


## Provenance integrity validation — SWA-013

SWA-012 established that disagreement can remain independently replayable. SWA-013 adds a stricter invariant: **the provenance graph itself must be structurally credible before its claims are consumed**.

The fixture introduces a deterministic integrity validator that checks:

- node identity uniqueness;
- unresolved edge targets;
- duplicate edges;
- dependency-key revision conflicts;
- self-derivation;
- cycles in the `DerivedFrom` graph;
- witness references that do not resolve to witness nodes;
- claim completeness certificates that disagree with the validator's observed closure;
- missing required claim dependencies.

Normalization is deterministic: node identities are sorted and duplicate identities are never silently merged; edges are sorted by source, target, and relation. This means two syntactically different orderings of the same valid graph produce the same normalized representation.

The validator intentionally distinguishes **structure from authority**. A graph can be structurally valid without authorizing an intervention, operating a building, or overriding a human decision. Likewise, a `Contradicts` edge is not treated as a derivation edge, so epistemic disagreement does not manufacture a causal cycle.

The intended result is:

`raw provenance -> normalized provenance -> integrity constraints -> Valid / Invalid`

The validator is deliberately scoped to the SWA fixture rather than claiming to be a complete implementation of W3C PROV. That boundary matters. W3C PROV defines validity in terms of normalization plus uniqueness, ordering, type, and impossibility constraints, and explicitly notes that cyclic derivation can imply an impossible history. SWA-013 adopts the same engineering direction while implementing only the invariants required by this reference architecture. citeturn0search0turn0search4

This changes the architecture from:

`claim -> provenance slice -> completeness certificate -> replay witness -> support/qualification/contradiction set`

to:

`claim -> provenance slice -> completeness certificate -> integrity validation -> replay witness -> support/qualification/contradiction set`

The next seam is **invalidation propagation**: when an upstream model, parameter, solver, dataset, or context revision changes, the system should deterministically propagate that dependency change to affected evidence, claims, replay witnesses, counterevidence branches, and decision contexts without mutating historical records.


## Deterministic invalidation propagation — SWA-014

SWA-014 turns the SWA-007 revalidation frontier into an explicit propagation record. A dependency revision is compared against the immutable historical dependency manifest of each downstream node. Only dependencies inside that node's declared closure can produce `RequiresRevalidation`.

The propagation layer covers the next provenance surfaces:

- validation evidence;
- claims;
- reproducibility witnesses;
- counterevidence branches;
- decision contexts.

The important semantic distinction is **revalidation is not historical deletion**. If model revision `m7` becomes `m8`, a claim produced under `m7` remains a historical claim. Its current status changes to `RequiresRevalidation` only when its declared dependency closure includes the changed model. A counterevidence branch is not erased or automatically resolved, and a decision context is only affected when its closure explicitly includes the changed dependency.

An incomplete historical dependency manifest produces `Unknown`, not `Current` and not `RequiresRevalidation`. This prevents missing provenance from masquerading as freshness.

The resulting flow is:

`dependency revision -> closure intersection -> deterministic propagation frontier -> downstream revalidation`

This is consistent with W3C PROV's model of revision as a specialized derivation and invalidation as a distinct lifecycle event: provenance records what happened historically while the current system can determine whether an entity remains usable under a changed dependency context. citeturn0search0turn0search1

SWA-014 deliberately does not convert stale evidence into false evidence. It also does not make a provenance result an authorization signal. The next useful seam is to connect this propagation frontier to the actual intervention/decision lineage so that a changed model can identify exactly which pending decision contexts require re-review without silently re-authorizing or rejecting anything.

## Decision revalidation boundary — SWA-015

SWA-014 already carried `DecisionContext` as a propagation node, so a separate generic decision model would duplicate existing work. SWA-015 instead makes the missing boundary explicit: **which pending decision contexts must be reopened when their required epistemic inputs are no longer current?**

The fixture consumes the SWA-014-style freshness frontier and a decision context's explicit evidence dependencies. It produces only:

- `Current` — every required evidence reference remains usable;
- `ReopenForReview` — at least one required evidence item requires revalidation;
- `Unknown` — required evidence is missing, provenance is unknown, or historical identity cannot be trusted.

It intentionally does **not** produce `Approve`, `Reject`, `Execute`, or any other authorization result.

This gives the architecture a clean separation:

`dependency change -> invalidation propagation -> evidence/witness freshness -> decision-context revalidation -> human/governance authorization`

The distinction is important:

- **ReopenForReview != Reject**. A changed model means the prior decision context must be reconsidered; it does not establish that the intervention is wrong.
- **RequiresRevalidation != False**. Staleness is an applicability state, not a truth judgment about the historical result.
- **Unknown != Current**. Missing provenance fails closed.
- **Historical authorization is immutable**. Revalidation creates a current review requirement without rewriting what was authorized historically.
- **Contradiction remains unresolved**. The boundary consumes the freshness state of evidence/witnesses; it does not choose between competing epistemic branches.

The fixture also canonicalizes evidence identifiers and dependency changes, so equivalent input orderings produce identical revalidation results.

This boundary fits Holochain's agent-centric validation model as a separate layer rather than conflating the two. Holochain validation is designed to deterministically validate whether authored operations conform to application rules, with dependencies explicitly addressable and unresolved dependencies yielding an indeterminate result rather than an implicit pass. Its source chains are append-only records of an agent's authored state changes. Those properties are useful for the future Mycelix receipt/governance layer, while SWA-015 remains a local deterministic epistemic fixture and does not pretend that a provenance record itself carries physical authority. citeturn0search0turn0search2turn0search4

The current Sol Atlas chain is therefore:

`claim -> provenance slice -> completeness certificate -> integrity validation -> replay witness -> support/qualification/contradiction set -> invalidation propagation -> decision revalidation boundary`

The next seam should not be another generic decision type. It should connect this boundary to the **intervention lifecycle** already represented elsewhere in the engineering fixture: preserve the proposed intervention, its counterfactual alternatives, expected outcomes, and reversibility/agency dimensions while making revalidation a prerequisite for a fresh authorization context. Mycelix/Holochain can then remain the durable distributed provenance/governance substrate rather than becoming a second physics or decision engine.


## Intervention identity binding — SWA-016

SWA-015 closes the evidence-freshness boundary, but one subtle gap remained: **fresh evidence does not prove that the intervention currently being considered is the intervention that was reviewed**.

SWA-016 binds the authorization context to:

- an exact intervention identifier and revision;
- an exact scenario identifier and revision;
- evidence explicitly bound to those same intervention/scenario revisions.

Therefore a changed intervention cannot inherit an otherwise-current evidence set merely because its human-readable name is unchanged.

The resulting boundary is:

`intervention revision + scenario revision + evidence freshness -> authorization-context readiness -> external authorization`

Readiness remains deliberately non-authoritative:

- `Current` means the context is internally coherent with its declared inputs.
- `ReopenForReview` means the intervention, scenario, or required evidence changed relative to what was reviewed.
- `Unknown` means required evidence cannot establish the binding.
- No state means execute, approve, reject, or physically actuate.

This closes an important time-of-check/time-of-use seam. A decision cannot silently migrate from intervention revision `v1` to `v2` while retaining the epistemic lineage of `v1`.

The architecture is now:

## End-to-end intervention revalidation — SWA-017

SWA-017 composes the contracts from SWA-011 through SWA-016 into one deterministic lifecycle fixture. Its purpose is to verify that the boundaries hold when evidence, intervention identity, model revisions, and historical authorization change together—not merely when each is tested in isolation.

The scenario follows an intervention at revision 1 with supporting, qualifying, and contradicting evidence. A model revision makes the old evidence require revalidation while preserving its historical identity and the prior authorization record. A revised intervention cannot inherit that evidence. A new review context must bind to the exact revised intervention and scenario, and new evidence must explicitly bind to those same revisions.

The lifecycle distinguishes:

- **Current** — the required evidence is present and bound to the exact intervention/scenario revisions;
- **ReopenForReview** — a required revision or evidence binding changed;
- **Unknown** — required evidence is absent or its status cannot be established.

The fixture retains contradictory and qualifying evidence as separate evidence identities; revalidation does not resolve the disagreement. It emits no aggregate score or intervention ranking. It also contains no physical actuation operation: readiness is an input to an external human or governance process, not a command to a building.

The end-to-end contract is:

`intervention/scenario revision -> evidence binding -> review readiness -> historical authorization -> dependency invalidation -> re-review -> new context -> fresh evidence -> external authorization`

This is a deterministic local reference scenario, not a Holochain validation zome or a Mycelix receipt implementation. Holochain's validation callbacks are expected to be deterministic for a given operation, and missing addressable dependencies yield an unresolved/indeterminate result rather than an implicit pass. That is a useful design constraint for a future Mycelix receipt adapter, but the distributed persistence and governance integration should reuse existing Mycelix zomes after their actual schemas and validation contracts are inspected. See the Holochain documentation on [validation](https://developer.holochain.org/build/validation/) and [validate callbacks](https://developer.holochain.org/build/validate-callback/).

SWA-017 therefore closes the local lifecycle composition seam. The next integration step is a narrow Mycelix/Holochain receipt projection with explicit content identity, author/authority separation, deterministic validation dependencies, and unresolved-dependency handling—not a second DKG, physics engine, or authorization system.

## Mycelix receipt boundary contract — SWA-018

SWA-017 established the local end-to-end review lifecycle. SWA-018 audits the existing public Mycelix domain surfaces and adds a local contract fixture for projecting a Sol Atlas review into Mycelix without inventing a parallel distributed record system.

The inspected Mycelix repository already contains distinct domains relevant to this vertical slice:

- **Attribution usage** defines voluntary dependency-usage receipts and usage attestations. These describe usage of a dependency; they are not generic prediction, validation, or authorization receipts.
- **Governance proposals and execution** represent collective requests and governance execution/timelock states. A proposal is not itself an adopted resolution or physical outcome.
- **Commons housing governance** defines cooperative meetings and resolutions.
- **Commons housing maintenance** defines maintenance requests, work orders, and inspections. These record operational workflow and observations, not a blanket authorization to change building controls.

SWA-018 therefore models an explicit projection envelope with a source namespace/object/revision, author reference, subject reference, declared content digest, disclosure class, exact dependency references, and separately optional authority/outcome references. Its example uses a governance-proposal target for a review request. It does not write to Holochain or claim to implement any Mycelix zome schema.

The contract checks that required identities and digest declarations are present, dependency references are unique and non-empty, and the evidence projection cannot be mislabeled as an Attribution usage receipt. Missing declared dependencies map to `UnresolvedDependencies`, not `Valid`. This mirrors Holochain's documented validation contract: deterministic validation may return valid, invalid, or unresolved dependencies; a missing `must_get_*` dependency should be retried rather than silently accepted. See [Holochain validate callbacks](https://developer.holochain.org/build/validate-callback/) and [must_get host functions](https://developer.holochain.org/build/must-get-host-functions/).

The key separation is:

`Sol Atlas evidence/provenance -> typed Mycelix projection -> deterministic DHT validation -> separate governance authorization -> separately observed physical outcome`

A valid projection is only structurally admissible under this fixture's local contract. It does not prove that a decision was authorized, that an intervention occurred, or that a building is safe. A Holochain record's author/signature and the domain's authority reference remain distinct; a receipt must never manufacture authority or substitute for physical telemetry.

**Integration caveat from inspection:** the Mycelix Attribution usage coordinator currently documents a graceful-allow path when the registry lookup is unavailable, while its usage integrity source itself flags a UsageAttestation update-validation gap where coordinator-only field restrictions are not sufficient against direct DHT operations. These are concrete review items before using that surface as a security boundary. They are not modified by SWA-018. The Mycelix README also characterizes the broader repository as pre-alpha and says multi-agent validation coverage varies by cluster, so a unit-tested projection fixture must not be described as end-to-end DHT validation.

SWA-018 closes the *mapping contract* seam only. The next step should be a focused Mycelix-side issue/patch proposal after inspecting the exact hApp DNA wiring and cross-zome identity conventions: select one destination record type, bind immutable source content identity and author, require addressable dependencies in integrity validation, preserve unresolved-dependency behavior, and add multi-agent tests for missing dependencies, tampered revisions, replay, and authorization separation. No DKG or chain-based replacement is warranted by this integration.

The fixture's digest value is intentionally marked as a non-cryptographic placeholder. SWA-018 checks that a digest declaration is present; it does not compute, verify, sign, or claim a content commitment. A production adapter must canonicalize the exact serialized payload and use an approved cryptographic digest/signature implementation before treating the value as an integrity commitment.


## Mycelix semantic-reference binding — SWA-019

SWA-018 established the local receipt boundary. SWA-019 tightens the interoperability seam against the **observed** Mycelix interoperability candidate at commit `b55bc03d99d0e8c89201dca06a264d16d5e2efd6`, which defines transport-neutral `SchemaRef` / `SemanticRef` primitives in the candidate cross-repository interoperability layer.

The Symthaea fixture does **not** import that repository or create a second authoritative semantic-reference implementation. It defines an adapter DTO whose fields deliberately mirror the observed transport shape:

`namespace + schema name + schema version -> opaque object id + optional object version`

The adapter then binds:

- a Sol Atlas source semantic reference;
- a distinct Mycelix governance-proposal target reference;
- a separate author reference;
- explicitly addressable dependency references;
- the observed Mycelix source revision as provenance;
- a separately declared authority reference, which is absent in the review fixture;
- a digest declaration whose status remains explicitly non-cryptographic until a production adapter defines canonical bytes and approved cryptographic binding.

SWA-019 adds deterministic checks for:

- schema identity changing when namespace, name, or version changes;
- source and target semantic identities remaining distinct;
- duplicate dependency rejection;
- missing required dependencies becoming `UnresolvedDependencies`;
- authority references not aliasing source or target identities;
- source-repository revision provenance not being interpreted as authority;
- replay producing identical serialized projection bytes.

This makes the integration chain more precise:

`Sol Atlas evidence/provenance -> semantic source reference -> typed Mycelix projection -> addressable dependency validation -> separate governance authorization -> separately observed physical outcome`

The observed Mycelix interoperability primitive itself is intentionally not treated as proof of a live deployed DNA, and the referenced commit is not treated as current runtime authority. Holochain validation remains the relevant distributed boundary: validation must be deterministic, dependencies must be addressable, and unavailable dependencies produce an unresolved result rather than an implicit pass. citeturn0search0turn0search1

### Engineering correction discovered during SWA-019 review

While inspecting the existing SWA-004 thermal counterfactual, the fixture contained a stale `hvac_max_kw` field in several `ZoneParameters` literals even though that field is not part of the struct and the intervention already owns HVAC capacity. SWA-019 work therefore also repaired SWA-004 by removing the invalid field and adding finite/positive input guards around the deterministic thermal step and comfort-band calculation.

This is a source-level correction; hosted CI/test execution is still not claimed unless GitHub reports it for the exact new head.


## Existing housing-governance lifecycle binding — SWA-020

The Mycelix inspection found that the Commons DNA already includes `housing_governance_integrity` and its coordinator, alongside `housing_maintenance` and the unified `commons_bridge`. The housing governance model contains an existing `Resolution` record with explicit proposer, meeting, voting, quorum, passed, and effective-date fields. The Commons DNA manifest wires the housing governance integrity and coordinator zomes rather than requiring a new Sol Atlas governance system.

SWA-020 therefore binds Sol Atlas to that **existing** lifecycle instead of inventing another proposal/authorization record.

The fixture distinguishes:

- `ReviewRequested` — Sol Atlas has produced a review request;
- `ResolutionObserved` — a governance resolution exists but adoption has not been independently established;
- `Adopted` — the observed resolution declares passed + quorum met + effective date.

Even `Adopted` does not automatically become local physical authority. The fixture requires an explicit external authority reference before reporting `ExternallyAuthorized`, and it contains no actuator, device command, or execution operation.

### Important Mycelix audit finding

The existing housing-governance integrity code itself documents a remaining vote-integrity gap: `vote_on_resolution` accepts caller-supplied aggregate vote counts, quorum, and the resulting passed state. The integrity layer constrains which fields may change but cannot independently establish that those aggregate numbers represent individual votes.

That means Sol Atlas should **not** treat `passed == true` as sufficient physical authorization merely because the field is present. A production adapter should bind to independently validated governance evidence—ideally the existing per-voter mechanisms where available—or require an explicit governance authority record whose own integrity rules establish the decision.

This is exactly the kind of boundary Holochain's validation model is designed to preserve: validation is deterministic and dependency-addressable, while unavailable dependencies remain unresolved rather than becoming implicit approval. citeturn0search0turn0search1turn0search3

The architecture is now:

`Sol Atlas intervention + evidence -> semantic reference -> Mycelix review projection -> existing housing governance resolution -> independently validated authority -> separately observed maintenance/physical outcome`

This is materially better than projecting directly into a generic "authorization receipt": the distributed system's existing domain semantics remain authoritative, while Sol Atlas remains responsible for evidence, provenance, model replay, and intervention identity.


### SWA-019 refinement: reuse the actual Mycelix semantic-reference contract

A deeper inspection found that the referenced Mycelix interoperability layer is not merely a proposal-shaped concept: at the inspected revision it already contains concrete `SchemaRef` and `SemanticRef` Rust types in `crates/mycelix-core-types/src/interoperability.rs`.

Those types explicitly establish:

- namespace + schema name + schema version as schema identity;
- opaque object identifiers;
- optional object-local versions;
- rejection of empty components;
- rejection of surrounding whitespace;
- rejection of control characters;
- bounded wire lengths;
- no implicit claims about authenticity, trust, authority, semantic equivalence, evidence quality, verification, or content binding.

SWA-019 now mirrors those **validation invariants** in its local adapter fixture rather than merely mirroring field names. This is important because an interoperability boundary is weaker if two systems agree on structure but disagree on what constitutes a valid identifier.

The adapter still does not duplicate the Mycelix implementation. The actual Mycelix types remain authoritative once a production adapter can depend on the appropriate published/workspace contract. The Symthaea fixture exists to make the boundary deterministic and testable without introducing a cross-repository runtime dependency into this engineering reference fixture.

This also reinforces the semantic rule already present in Mycelix's interoperability documentation: overlapping labels do not imply schema equivalence. A consumer must retain namespace, schema identity, and version rather than collapsing them into a shared ordinal or display label.


## Content binding — SWA-021

SWA-019 established semantic identity and dependency structure, but its digest was intentionally only a declaration. SWA-021 closes that gap at the local fixture boundary without pretending to be the final Mycelix cryptographic contract.

The projection now derives a BLAKE3-256 digest from a deterministic serialized `ContentBindingPayload` containing the schema, an explicit evidence-binding manifest, source/target semantic references, projection target, observed Mycelix revision, author/authority references, dependency set, and authority/actuation state. The digest is computed over the payload **excluding the digest field itself**, avoiding a circular commitment. BLAKE3 provides a 32-byte default hash and deterministic hexadecimal representation in its Rust implementation. citeturn1search0turn1search1

This establishes three useful invariants:

- identical semantic input produces the same digest;
- changing a committed field changes the digest;
- the digest cannot silently become authority, authorization, or proof of physical execution.

The algorithm is explicitly labeled in the envelope. This is a **fixture-level content commitment**, not yet a claim that Mycelix accepts BLAKE3 as its production receipt algorithm. The eventual adapter must use the target Mycelix crypto/profile contract and canonical wire representation rather than allowing each integration to invent its own commitment format.

The resulting boundary is now:

`Sol Atlas evidence identity -> semantic references -> canonical projection payload -> cryptographic content commitment -> Mycelix validation/governance boundary -> separately observed physical outcome`


## Evidence-slice binding — SWA-022

The next trust-boundary refinement is now explicit: a valid projection must identify the exact Sol Atlas evidence slice it claims to represent.

SWA-022 adds an `EvidenceBinding` manifest containing semantic references for the provenance slice, claim, evidence, model, scenario, and dataset, plus the slice revision. The manifest receives its own BLAKE3-256 digest, and that digest is then included in the outer projection content commitment.

This is deliberately a **manifest commitment**, not a second provenance implementation. The authoritative provenance graph remains owned by the existing SWA provenance fixtures. The adapter consumes the slice identity/revision contract and commits to the exact dependency set presented to Mycelix.

That distinction matters:

- a valid outer receipt can no longer silently point at a different model/scenario/dataset revision without changing its content commitment;
- the evidence slice has an explicit identity rather than being represented only by a generic `evidence` dependency;
- provenance integrity and Mycelix interoperability remain separate concerns;
- the adapter still does not claim that a digest proves the underlying physical world state.

Holochain's validation model strongly favors this structure: dependencies used for validation need deterministic, addressable identities, and unavailable dependencies are unresolved rather than silently accepted. citeturn0search0turn0search2

The intended production evolution is therefore **not** to duplicate the provenance graph inside the Mycelix adapter. Instead, the Sol Atlas provenance layer should export a canonical slice identifier, exact dependency revisions, and an approved content commitment; the Mycelix adapter binds those references into its projection.


## Evidence-slice canonicalization — SWA-023

SWA-022 exposed an important precision gap: a manifest commitment is only useful if the manifest is itself canonicalized from the provenance slice rather than assembled as an adapter-local list.

SWA-023 moves that canonicalization primitive into the `symthaea-engineering` library as `provenance_binding::EvidenceSliceManifest`. It is intentionally narrower than a provenance graph:

- the provenance graph remains the semantic source of truth;
- the binding layer accepts the already-selected slice members and relationships;
- node and edge ordering is canonicalized before serialization;
- node identity, kind, revision, edge endpoints, and edge relation are all committed;
- changing a dependency revision or edge relation changes the digest;
- removing contradictory evidence changes the digest rather than allowing it to disappear into an aggregate result.

The SWA-019 Mycelix fixture now binds its evidence digest to the complete reference slice represented by SWA-009's current fixture topology: claim, evidence, prediction, model, parameters, scenario, dataset, and context, including their directed provenance relationships.

This closes the main weakness in SWA-022 without creating a second provenance system. The production path should eventually have the provenance exporter construct the `EvidenceSliceManifest` mechanically from the authoritative graph, so the Mycelix adapter never interprets or reconstructs provenance semantics itself.

The trust boundary is now:

`authoritative provenance graph -> canonical evidence-slice manifest -> BLAKE3-256 commitment -> semantic Mycelix projection -> Mycelix validation/governance -> independently observed outcome`

This also matches Holochain's validation model: validation should be deterministic, dependencies should be addressable, and unavailable dependencies should remain unresolved rather than becoming implicit approval. citeturn1search0turn1search2


## Canonical encoding hardening — SWA-024

SWA-024 hardens the byte-level identity beneath SWA-023.

The evidence-slice commitment no longer depends on JSON serialization. The canonical byte representation is now explicitly versioned and length-delimited:

- domain/version separator: `symthaea:evidence-slice:v1`;
- length-prefixed scalar strings;
- explicit presence markers for optional revisions;
- canonicalized node and edge ordering;
- explicit collection lengths;
- BLAKE3-256 over those bytes.

This removes an unnecessary dependency between the cryptographic identity and a presentation/wire serialization format. It also makes representation boundaries explicit: an absent revision and an explicitly empty revision are different byte sequences.

BLAKE3 itself is unchanged; the improvement is the **input domain and framing** being committed. The BLAKE3 specification defines domain-separated modes and a fixed 256-bit default hash output; the fixture's own domain/version separator prevents unrelated Symthaea content from accidentally sharing this commitment namespace. citeturn0search4turn0search6

This remains a content commitment, not a signature and not authorization. A production Mycelix adapter will still need the target system's accepted cryptographic/profile contract.
