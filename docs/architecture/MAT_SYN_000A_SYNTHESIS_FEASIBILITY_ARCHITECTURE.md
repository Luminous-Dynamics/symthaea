# MAT-SYN-000A — Evidence-bounded materials synthesis feasibility architecture

Status: architecture freeze candidate

Authority: representation / composition only; no physical execution authority

Base: current `main` at `eae17187e199e3a53d108b437c0215b5ff812261`

## 1. Purpose

Symthaea already has substantial materials, experiment, magnetics, evidence, and manufacturing semantics. The missing boundary is not another generic materials ontology and not another generic process engine. It is a narrow composition theorem for answering a more disciplined question:

> What exact evidence supports that a material candidate has at least one plausible, testable synthesis-and-processing route under a declared context?

This document freezes that boundary before any production `MAT-SYN` Rust implementation is allowed.

The architecture is deliberately fail-closed and non-scalar. It must never turn thermodynamic plausibility, a generated route, a valid process plan, an equipment capability claim, or one successful physical batch into a universal `synthesizable=true` fact.

## 2. Core theorem

```text
candidate proposed
!= thermodynamically supported
!= synthesis route hypothesized
!= route physically plausible
!= process plan represented
!= resources capable
!= resources available
!= physical trial authorized
!= physical trial executed
!= target phase observed
!= target property validated
!= route repeatable
!= route robust
!= manufacturable
!= economically scalable
```

Additional hard boundaries:

```text
negative formation energy
!= convex-hull stability
!= dynamical stability
!= kinetic accessibility
!= synthesizability
```

```text
DFT support
!= experimental synthesis evidence
```

```text
recipe exists
!= recipe qualified
!= recipe executable here
!= recipe executed
```

```text
one successful batch
!= repeatable synthesis
!= independent replication
```

```text
target phase observed
!= target functional property established
```

```text
intrinsic magnetic quality
!= coercivity
!= bulk permanent-magnet performance
```

## 3. Architectural decision: composition first, zero duplicate common types

The default type budget for shared production semantics in MAT-SYN is:

```text
new generic materials/process/provenance/execution types = 0
```

A future new public type is allowed only if an executable semantic-gap fixture proves that the required proposition cannot be represented losslessly by existing owners plus exact references.

MAT-SYN should be a composition/projection layer over canonical owners, not a second owner of their semantics.

## 4. Canonical owners to reuse

### 4.1 Materials scientific authority

Reuse a current qualified successor of the MAT evidence stack for:

- evidence authority progression;
- external import vs local reproduction vs surrogate prediction vs physical measurement;
- exact scientific evaluation identity;
- applicability / out-of-domain state;
- uncertainty and calibration;
- computational evidence admission;
- thermodynamic property semantics.

Historical examples include MAT-001, MAT-002, MAT-011/MAT-011B and MAT-016A/B. Their historical presence does not make them current qualified dependencies.

MAT-SYN must not copy their property values, uncertainty types, authority states, or evaluator identities.

### 4.2 Physical sample and experiment lineage

Reuse a current qualified successor of MAT-013 semantics for:

- precursor-lot identity;
- synthesis/process protocol bindings;
- batch/sample lineage;
- reprocessing, splitting and merging;
- characterization runs;
- calibration/raw/derived artifact bindings;
- physical-lineage independence.

MAT-SYN must not create another sample DAG.

### 4.3 Materials experiment planning

Reuse a current qualified successor of MAT-014 semantics for:

- candidate next-action proposals;
- acquisition strategy;
- campaign budget accounting;
- protocol-bound human authorization;
- deliberate absence of direct physical execution authority.

MAT-SYN may make evidence available to planning. It must not become an experiment executor.

### 4.4 Manufacturing process semantics

Reuse MFG-PROC for:

- process-family/profile identity;
- input/output/preserved process-state references;
- capability requirements and capability evidence;
- immutable recipe identity/commitments;
- process-plan DAGs;
- bounded rework;
- capability history and yield evidence.

Current-main replay candidate #5902 is the relevant public-contract direction, but no dependency may be treated as qualified until its exact executable qualification is successful.

MAT-SYN must not create another process registry, recipe format, capability matcher, process DAG, or rework graph.

### 4.5 Magnet-specific semantics

Reuse a current qualified successor of MAG-002 for magnet-specific evidence such as:

- intrinsic vs extrinsic magnetic-property separation;
- microstructure context;
- process state;
- phase fractions;
- grain size;
- misorientation;
- porosity;
- interface density;
- coercivity/remanence/(BH)max evidence context.

MAT-SYN must not infer coercivity from K1 or magnetization.

### 4.6 Observation, safety, independence, and currentness

Reuse canonical owners rather than inventing local substitutes for:

- calibrated physical observation;
- units/quantities/conditions;
- hazards and safety review;
- source/evidence independence and common-mode dependence;
- currentness/validity;
- actual operational resource/lot/custody facts.

When a canonical owner is absent or unqualified, MAT-SYN must preserve that gap rather than filling it with a local convenience field.

## 5. Synthesis feasibility is route- and context-relative

A material should not receive a context-free binary `synthesizable` property.

The assessment subject is conceptually:

```text
material candidate
+ target phase / structure
+ declared route hypothesis
+ declared process context
+ declared scale
+ declared observation / acceptance criteria
+ exact evidence snapshot
```

A different precursor set, processing route, thermal history, atmosphere, pressure regime, scale, equipment envelope, or acceptance criterion is a different proposition.

## 6. Required evidence vector

A future MAT-SYN projection should expose independent dimensions rather than an aggregate score.

### 6.1 Target scientific state

References should establish, where applicable:

- exact composition / structure identity;
- thermodynamic evidence;
- phase-competition evidence;
- dynamical-stability evidence;
- applicability/OOD state;
- relevant target-property evidence.

Missing evidence remains missing.

### 6.2 Route hypothesis

A route hypothesis must retain an exact reference to its scientific/engineering rationale.

Examples of rationale classes may include:

- literature-supported route;
- phase-diagram / thermodynamic rationale;
- kinetic/pathway model support;
- analogous-material transfer hypothesis;
- exploratory human hypothesis.

These are evidence descriptions, not interchangeable authority levels.

```text
analogy
!= demonstrated pathway
```

### 6.3 Precursors and starting state

The route must reference exact precursor/material-state requirements rather than only chemical names.

Where physical material exists, actual lot identity remains owned by the physical-lineage/provenance owner.

```text
precursor nominally purchasable
!= precursor lot acquired
!= precursor purity verified
```

### 6.4 Process representation

The route should resolve to exact MFG-PROC process definitions / transformation contracts / recipe commitments / process-plan refs as appropriate.

A prose route may remain a hypothesis, but it cannot silently become a represented process plan.

### 6.5 Capability evidence

Every capability-critical step should retain exact capability requirement and capability-evidence refs.

Distinct states must remain distinct:

```text
process representable
!= resource claims capability
!= capability evidence sufficient
!= capability evidence current
!= resource available
```

### 6.6 Environment and boundary conditions

The evidence snapshot should bind the conditions whose change can invalidate route support, including when relevant:

- atmosphere;
- pressure regime;
- temperature regime;
- contamination/cleanliness boundary;
- precursor state;
- geometry/scale;
- thermal history;
- measurement environment.

MAT-SYN should reference canonical quantity/condition owners rather than copy numerical semantics.

### 6.7 Hazard / safety review

A plausible route can still be unsafe or unauthorized.

```text
scientifically plausible
!= safe
!= authorized
```

MAT-SYN may require exact safety-profile/review refs before a planning projection can call a route trial-ready under declared policy. It must not itself make safety determinations or authorize execution.

### 6.8 Observation and acceptance plan

Before a physical trial is interpreted, the route should declare what evidence would discriminate success from failure.

At minimum the plan should separate:

- process completion;
- target-phase/structure observation;
- composition/contamination observation;
- microstructure observation when performance depends on it;
- target-property characterization;
- null/failed/instrument-fault dispositions.

```text
process completed
!= target phase observed
```

### 6.9 Physical trial evidence

Executed synthesis evidence must bind an exact physical lineage and observation set.

A successful process receipt alone cannot establish synthesis success.

A target phase should be supported by characterization evidence under declared acceptance criteria.

A target property should require its own characterization evidence.

### 6.10 Repeatability and independence

Repeated observations on the same lineage are not independent synthesis replications.

Sibling specimens split from one batch cannot satisfy independent-batch replication by themselves.

A stronger route claim should retain:

- number of physical root batches;
- exact independence/common-mode evidence;
- process/precursor/equipment context;
- negative/null outcomes as well as successes.

No ratio should silently erase missing or censored outcomes.

### 6.11 Currentness and drift

A historically successful route is not automatically a current executable route.

Material changes may include:

- precursor source/purity drift;
- process/recipe revision;
- equipment/configuration change;
- calibration change;
- environment change;
- acceptance-criterion change;
- safety-policy change;
- target structure/property definition change.

Historical success must remain historical evidence; current route support should be separately re-evaluated.

## 7. No universal synthesizability score

Forbidden production concepts include:

```text
synthesizability_score
manufacturability_score
route_confidence_score
readiness_score
best_route_score
```

A decision layer may rank experiments using an explicitly declared campaign objective, cost model, or value-of-information policy. Such ranking remains a decision projection and cannot become scientific evidence authority.

## 8. Suggested derived dispositions

If a future adapter needs compact UI/query dispositions, they should be lossy projections over the full vector and must retain links to the exact evidence snapshot.

Possible non-authoritative dispositions:

```text
EvidenceInsufficient
RouteHypothesisRepresented
RouteEvidenceConflicting
RoutePreconditionsPartiallySupported
RoutePreconditionsSupportedForPlanning
PhysicalTrialObserved
TargetPhaseObserved
TargetPropertyObserved
RepeatabilityEvidencePresent
RequalificationRequired
```

These dispositions must not replace the underlying evidence dimensions.

In particular:

```text
RoutePreconditionsSupportedForPlanning
!= trial authorization
!= execution
!= synthesis success
```

## 9. Fail-closed rules

A future executable projection must fail closed when any required strong claim depends on:

- unresolved subject identity;
- stale or unknown applicability evidence;
- an OOD computational result being treated as in-domain;
- unknown process capability where capability is required;
- stale capability evidence;
- unresolved process-state references;
- missing required route step;
- missing precursor identity for an executed physical trial;
- missing physical lineage;
- missing target-phase characterization;
- missing microstructure characterization for an explicitly microstructure-dependent performance claim;
- missing target-property characterization;
- contradictory evidence that has not been dispositioned;
- missing required safety/authorization reference for a proposed physical action;
- identity-changing drift since the evidence snapshot.

Unknown must never mean satisfied.

## 10. Adversarial corpus requirements

Before Rust production code, freeze a synthetic corpus containing at least these cases.

### Positive / bounded-support cases

1. computationally supported phase + represented route + current capability refs, but no physical trial;
2. physical trial with target phase observed, target property not yet tested;
3. target phase + property observed in one root batch, with no independent repeatability claim;
4. two genuinely independent root batches with consistent route/phase observations;
5. changed equipment configuration explicitly shown non-material by an external canonical owner, retaining bounded route support.

### Near-miss / rejection cases

6. negative formation energy presented as `synthesizable`;
7. low hull distance presented as a demonstrated route;
8. generated prose recipe presented as an executable process plan;
9. valid process plan with no capability evidence;
10. capable resource with stale capability evidence;
11. capable resource but no operational availability evidence;
12. physical execution receipt with no phase characterization;
13. target phase observed but property claim promoted without property characterization;
14. intrinsic K1 result promoted to coercivity;
15. coercivity claim with missing microstructure context;
16. two sibling coupons promoted to independent synthesis replication;
17. one successful batch with earlier failed batches omitted;
18. changed precursor lot reusing the old sample identity;
19. route/recipe mutation reusing prior authorization;
20. stale historical route success promoted to current route readiness;
21. OOD surrogate promoted to scientific route support;
22. external database value recast as local DFT reproduction;
23. unsupported safety state treated as physical authorization;
24. simulated synthesis promoted to physical synthesis evidence;
25. manufacturability/economic scale claim derived solely from coupon-scale synthesis success;
26. aggregate synthesizability score hiding a missing critical dimension.

Each fixture should declare exact expected vector consequences, not merely `accept/reject`.

## 11. Magnet specialization

Rare-earth-free Fe-Co-X remains an appropriate first proving vertical because it exposes the full separation between intrinsic material prediction and process-dependent bulk performance.

A magnet route should preserve at least the following conceptual chain:

```text
candidate composition / structure
        ↓
thermodynamic + structural evidence
        ↓
route hypothesis
        ↓
process plan / capability evidence
        ↓
physical batch lineage
        ↓
phase + composition characterization
        ↓
microstructure characterization
        ↓
intrinsic magnetic characterization
        ↓
extrinsic magnetic characterization
        ↓
repeatability / independent replication
```

For this vertical:

```text
high predicted K1
!= high coercivity
```

and:

```text
successful phase synthesis
!= useful permanent magnet
```

This is the correct bridge from the existing MAG-EXP retrospective evidence line into prospective physical discovery.

## 12. Relationship to MAG-EXP-002

MAG-EXP-002 is a retrospective evidence/execution subject. Even after its exact repaired source qualifies, it should establish only the propositions explicitly demonstrated by that experiment and its evidence receipts.

```text
retrospective corpus qualified
!= candidate synthesis route known
!= candidate physically synthesized
```

MAT-SYN must consume exact scientific evidence refs rather than infer synthesis feasibility from benchmark membership, ranking, or retrospective success.

## 13. Relationship to MAG-001 benchmark

MAG-001 already owns leakage-resistant rare-earth-free magnet benchmarking, including composition-family / structure-cluster / combined holdouts and sealed evaluator targets.

MAT-SYN must not create a second benchmark framework.

A benchmark can establish bounded predictive/search performance under a frozen split. It cannot establish physical synthesizability.

```text
OOD benchmark success
!= synthesis success
```

## 14. Relationship to future active learning

MAT-014 already provides the correct dry-run orchestration direction for experiment selection.

A future active-learning loop should therefore look conceptually like:

```text
MAT scientific evidence
+ MAG-001 leakage-safe benchmark evidence
+ MAT-SYN route-evidence projection
+ campaign objectives / uncertainty / information gain
        ↓
MAT-014 proposal
        ↓
explicit human authorization where physical
        ↓
external execution boundary
        ↓
MAT-013 physical lineage + observations
        ↓
new scientific evidence
```

No MAT-SYN component should directly actuate equipment.

## 15. Implementation sequence

Recommended bounded sequence:

### MAT-SYN-000A — this architecture

Freeze ownership, negative theorems, vector semantics, and implementation gate.

### MAT-SYN-001A — frozen synthetic composition corpus

Data/docs only. Encode the adversarial cases above using exact opaque references rather than reimplementing owner payloads.

### MAT-SYN-001B — independent reference validator

A small dependency-light validator that evaluates only the frozen composition semantics and proves the corpus has executable known answers.

### MAT-SYN-002A — minimal Rust projection adapter

Only after current qualified owner interfaces exist. Prefer zero new generic production types; if one materials-specific projection is necessary, justify every field by a semantic-gap fixture.

### MAG-SYN-001A — Fe-Co-X specialization

Bind magnet-specific phase/microstructure/property requirements through canonical MAG contracts without copying their semantics.

### MAT-SYN-003A — MAT-014 planning bridge

Expose route-evidence snapshots as inputs to experiment proposal generation while preserving proposal != authorization != execution.

### MAT-SYN-004A — prospective evidence loop

Preregister a small prospective campaign with sealed candidate/criterion identities, physical lineage, complete negative/null outcome retention, and independent replication rules.

## 16. Production implementation gate

No production MAT-SYN Rust code should open until all of the following are true:

1. the required materials scientific-evidence owner has a current, qualified public interface;
2. the required physical-lineage/characterization owner has a current, qualified public interface;
3. the required MFG-PROC public contracts have a current, qualified interface;
4. the synthetic MAT-SYN corpus is frozen with exact digest and known-answer semantics;
5. an independent reference validator executes that corpus successfully;
6. every proposed MAT-SYN production field has an explicit canonical owner or a demonstrated semantic-gap justification;
7. no field can manufacture physical/synthesis authority from planning, simulation, ranking, or external database evidence;
8. physical execution remains outside the MAT-SYN authority boundary.

## 17. Claim ceiling

This architecture may establish only that a future implementation has a disciplined composition target.

It establishes no:

- material stability;
- synthesis route validity;
- kinetic accessibility;
- process capability;
- recipe qualification;
- equipment availability;
- physical authorization;
- physical execution;
- synthesized material;
- target phase;
- magnetic property;
- repeatability;
- independent replication;
- manufacturability;
- economic viability;
- safety;
- regulatory compliance.

The intended end state is not an AI that says "this material is synthesizable."

It is an evidence system that can say, with exact references and bounded authority:

```text
Here is the candidate.
Here is what is computationally supported.
Here is the proposed route.
Here is what process/capability evidence exists.
Here is what is missing or conflicting.
Here is the physical lineage if a trial occurred.
Here is what phase/property evidence was actually observed.
Here is whether the result has been repeated independently.
Here is what changed since the evidence snapshot.
Here is the exact boundary beyond which we do not know.
```
