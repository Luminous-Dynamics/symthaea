# MFG-FOUNDATION-001 — Physical Technology Improvement Engine

Status: draft research architecture
Base subject: `458c7b98d81c64b9361e252f85ef9d45132e6682`

## Purpose

Define a reusable, evidence-bounded process for identifying and improving mature physical technologies whose progress is constrained less by missing fundamental physics than by fragmented evidence, poor lifecycle feedback, weak interoperability, expensive experimentation, maintenance burden, over-customization, or design-to-manufacturing disconnects.

This is a cross-domain research/orchestration layer. It is deliberately **not** a second materials ontology, measurement system, evidence-dependency graph, hypothesis ontology, manufacturing ledger, lifecycle graph, or authority system.

It does not authorize manufacturing, deployment, biological experimentation, maintenance, procurement, safety-critical control, or regulatory claims.

## Governing rule — reuse before abstraction

Every proposed common field must first answer:

```text
who already owns this fact?
```

If a canonical owner exists, MFG-FOUNDATION stores an exact typed reference instead of copying the value, identity, lineage, calibration, validity, provenance, lifecycle state, hypothesis, execution state, or authority.

The zero-duplication gate is tracked in #5768.

## Canonical ownership map

### Materials research

The repaired MAT line is intended to own:

- MAT-007 — physical material subject identity;
- MAT-008 — conditioned material/property evidence, methods and uncertainty;
- MAT-009 — campaign, hard constraints, independent objectives and Pareto semantics;
- MAT-010 — failed/null/duplicate/OOD SearchMemory;
- MAT-013 — physical sample/process lineage and scale;
- MAT-014 — proposal/acquisition/human-review discovery orchestration.

The MAT-RFMT recovery program #5323 is reconstructing the historical stack on current `main`. MFG-FOUNDATION must not bind implementation to stale historical branch interfaces.

### Quantities, units, frames and time

SE-SEM-001 #4861 owns canonical physical quantity-kind, unit, conversion, frame, shape and clock/time-base semantics.

Therefore this program must not create local physical-value standards such as:

```text
Pressure
Power
Flow
Temperature
CrackWidth
Torque
Duration
unit: String
```

merely for convenience.

### Physical observations and calibration

FIELD-000 #3589, FIELD-001A #3633 and SE-OBS #3695 own the common physical-observation, source/device/configuration, calibration/traceability, uncertainty, timestamp and raw-vs-derived evidence boundaries.

A MFG-FOUNDATION record should normally point to the authoritative observation rather than copying its measured payload.

### Evidence dependency, independence and replication

ENG-EVID-INDEP-001 #4918 owns domain-neutral evidence-dependency/common-mode lineage, claim-relative independence profiles, repeat-vs-replication distinctions, calibration/clock/model/implementation common modes and indeterminate-dependence semantics.

Therefore:

```text
multiple observations
!= multiple independent observations
```

and MFG-FOUNDATION must not create a second generic source-lineage or `LeakageGroupRef` ontology.

#5756 owns only benchmark partition policy and deterministic partition receipts over canonical dependency refs.

### Hypotheses and failure causes

Hypothesis identity stays with the domain that owns the proposition.

Examples:

- MAT-HYPOTHESIS-001 #5254 — materials mechanism/process/intervention hypotheses;
- SE-VV #3697 — engineering failure-mode/effect/cause and discrepancy-investigation hypotheses;
- domain-specific physics/research subjects where a more specific owner exists.

#5757 owns only the evidence-bearing attribution relation from an observed effect to an existing hypothesis/mechanism subject.

### Applicability, validity and currentness

A generic MFG `ApplicabilityProfile` is not approved.

Existing ownership/precedent includes domain study-scope profiles, SE-VV intended-use/validity envelopes, SE-SEMANTICS applicability-review consequences and ETK authority-bearing currentness/contradiction/supersession/applicability-loss semantics.

Until an exact generic-owner audit proves a real gap, MFG-FOUNDATION carries only an external validity/applicability-scope reference.

### Physical components and articles

ENG-CATALOG #5675 owns generic component/source-document/BOM/realized-article identity.

ROT-EQUIP should reference existing component/article subjects where possible instead of inventing a second catalog identity.

### Manufacturing and lifecycle

MFG-PROC owns process definition, process-state transformation, capability, recipe, process-plan, inspection/hold/rework and capability-history semantics.

MFG-LIFE owns lifecycle strategy/design/assessment semantics.

Mycelix owns concrete operational provenance for plan instances, work/lot/batch/unit scope, provider/site/resource assignments, manufacturing/service/lifecycle events, and biological material lots/containment references where applicable.

Preserve:

```text
process plan
!= plan executed

inspection planned
!= inspection executed
!= result accepted

historical capability
!= future capability guarantee
```

### Verification, reliability and discrepancy

SE-VV owns generic V&V, credibility, reliability/maintainability, failure-mode/cause and anomaly/discrepancy lifecycle semantics.

MFG-FOUNDATION may relate those facts to hypotheses; it should not create another generic FMEA/R&M system.

## Core separation

```text
technology constraint observed
!= validated intervention
!= mechanism established
!= manufacturable product
!= qualified production process
!= safe field deployment
```

Likewise:

```text
observation
!= admitted evidence
!= mechanism support
!= recommendation
!= human decision
!= authorization
!= execution
!= successful outcome
!= qualification/certification
```

## No `stagnant=true`

"Stagnation" is not a primitive truth value.

A candidate opportunity must instead reference concrete evidence such as:

- energy/conversion loss;
- material waste;
- failure/degradation burden;
- maintenance/downtime burden;
- interoperability fragmentation;
- excessive specification/variant fragmentation;
- poor repairability/remanufacturing;
- weak field feedback;
- unresolved mechanism;
- weak lab-to-field transfer;
- deployment/integration barriers;
- standardization opportunity;
- closed-loop-control opportunity.

These are descriptive constraint classes. They do not establish solvability, importance, priority, or commercial value.

## Minimal provisional common kernel

The current **upper bound**, not an implementation target, is approximately:

```text
TechnologyConstraintClassV1
TechnologyConstraintObservationV1
BenchmarkPartitionPolicyV1
BenchmarkPartitionReceiptV1
MechanismAttributionEdgeV1
```

Audit has already removed proposed common primitives:

```text
LeakageGroupRefV1
  -> ENG-EVID-INDEP #4918 already owns dependency/independence

MechanismHypothesisV1
  -> domain owners already own hypothesis identity

ApplicabilityProfileV1
  -> blocked pending proof that existing applicability/validity owners are insufficient
```

If further audit eliminates another type, remove it. Issue existence does not justify code.

## Technology constraint reference envelope

#5755 now treats the common observation as reference-only, conceptually:

```text
TechnologyConstraintObservationV1 {
  technology_subject_ref,
  constraint_class,
  evidence_ref,
  evidence_dependency_ref?,
  operating_context_ref,
  observation_period_or_time_ref?,
  applicability_or_validity_scope_ref?,
  contradiction_or_related_evidence_refs[],
}
```

This envelope does not own the measurement value, unit, calibration, asset/material identity, dependency graph, applicability ontology, or authority.

## Benchmark partitioning

#5756 consumes ENG-EVID-INDEP dependency/common-mode refs and adds only benchmark policy plus deterministic group-to-partition assignment receipts.

The grouping policy is claim-relative:

```text
leave-row-out
!= leave-study-out
!= leave-asset-out
!= calibration-independent evaluation
```

BIO-CEM study holdout should keep one paper, supplement, copied table and all formulations from that source dependency together.

ROT-EQUIP leave-asset-out should keep all relevant windows from one asset together.

A leakage-safe split is an evidence-integrity property only; it does not establish model quality or causal validity.

## Mechanism attribution

#5757 owns the relation, not the hypothesis itself.

Conceptually:

```text
MechanismAttributionEdgeV1 {
  effect_ref,
  mechanism_or_hypothesis_ref,
  applicability_or_validity_scope_ref,
  state,
  evidence_links[],
  competing_mechanism_refs[],
  confounder_refs[],
  limitations[],
}
```

Recommended bounded states:

- SupportedUnderProfile;
- ChallengedUnderProfile;
- Unresolved;
- NotApplicableUnderProfile.

Evidence roles may include support, challenge, control, confounder, replication, transfer and discrimination against alternatives.

```text
HDC similarity
!= shared mechanism

model prediction
!= observation

correlation
!= intervention effect

before/after improvement
!= causal attribution
```

## First proving domain — BIO-CEM

BIO-CEM uses #5753/#5767/#5759 to prove the architecture against a literature/materials-heavy domain.

Independent evidence planes remain:

```text
VisibleCrackClosure
TransportRecovery
MechanicalRecovery
MineralizationEvidence
BiologicalViability
DurabilityResponse
RepeatedHealingResponse
```

They are not one healing score.

Biological state is explicit and observation-relative; `living=true` is forbidden as a scientific shortcut.

BIO-CEM owns only genuinely domain-specific vocabulary such as biological observation state, healing output-plane classification, control roles, literature extraction state and method/comparability state. Material identity, measurements, source dependency, hypotheses, physical sample lineage, operational biological lots and authority remain with their canonical owners.

## Second proving domain — ROT-EQUIP

ROT-EQUIP #5760 uses a deliberately different system boundary:

```text
electrical supply
-> drive/starter
-> motor
-> coupling/gearbox
-> pump | fan | compressor
-> piping/ducting/valves/dampers/restrictions
-> process load
-> controls
```

Independent observation planes include electrical input, mechanical state, fluid/air process state, useful output/system performance, and maintenance/degradation evidence.

Preserve:

```text
component efficiency
!= system efficiency

anomaly detected
!= fault identified
!= root cause established

maintenance performed
!= restored performance

predicted savings
!= measured savings
```

ROT-EQUIP should add only its true domain vocabulary: narrowly defined topology/role semantics, operating-context binding and rotating/flow-system mechanism families that no existing owner supplies.

## Cross-domain generality gate

The program does not expand merely because another industry is interesting.

First prove:

```text
BIO-CEM
        \
         > same small reference / partition / attribution substrate
        /
ROT-EQUIP
```

without:

- `AnyValue`;
- free-form JSON semantic escape hatches;
- duplicate quantity/unit systems;
- duplicate observation/calibration systems;
- duplicate evidence-independence graphs;
- duplicate hypothesis ontologies;
- duplicate component identities;
- duplicate manufacturing/lifecycle ledgers;
- duplicate currentness or authority systems.

Only after that proof should industrial thermal systems, building controls, corrosion/coatings, membranes, transformers or other sectors become new adapters.

## Portfolio decision layer

#5758 projects explicit evidence into existing Pareto/decision-analysis machinery rather than implementing another optimization engine.

Preserve:

```text
opportunity evidence
!= criterion value
!= Pareto membership
!= scalar priority
!= selected project
!= funded/procured project
```

Missing evidence stays missing. Any scalar weighting is an explicit policy artifact with inspectable weights, normalization, missing-data policy and sensitivity; it never becomes scientific evidence.

## External constraint corpus

#5761 is a versioned evidence corpus/validator, not a new registry service.

Every public-source claim preserves exact population/context/time/source lineage and claim class. Historical evidence cannot silently become a current global fact, and newer sources do not automatically defeat older evidence.

## Intervention/outcome composition

#5762 binds proposal, external work/authority refs, external execution receipts, post-observation refs and #5757 attribution refs without creating another work-order or execution ledger.

```text
recommendation
!= authorization
!= execution
!= correct execution
!= improved performance
!= causal proof
```

## Research-to-manufacturing transfer

#5763 is a derived assessment over MAT sample evidence, MFG-PROC capability/history, FIELD/SE observations and Mycelix operational provenance.

Independent transfer dimensions replace a universal `manufacturable=true` or `ProductionReady` state.

```text
coupon result
!= batch capability
!= stable production process
!= field durability
```

## Negative evidence

#5764 projects null/failed/contradictory/OOD/replication/transfer outcomes into existing MAT-010, SE-VV, #5757 and operational owners rather than creating another negative-results ledger.

```text
null result
!= missing result

failed intervention
!= impossible intervention

replication disagreement
!= fraud
```

## Currentness and applicability

#5765 is a read-only projection over ETK/SE/domain currentness and applicability owners.

```text
old evidence
!= invalid evidence

recent evidence
!= superior evidence

applicable under profile A
!= applicable under profile B
```

## Authority non-escalation

#5766 freezes authority-composition rules only.

MFG-FOUNDATION artifacts normally grant zero physical authority.

```text
derived authority
<= explicitly referenced external authority facts
```

No quantity of confidence, simulation success, HDC similarity, provenance completeness, Pareto priority or supporting studies can manufacture missing permission.

## Type-budget gate

Every implementation PR beneath #5754 must include a table equivalent to:

```text
proposed field/type
semantic meaning
existing owners searched
canonical owner/ref used
why a genuinely new semantic is still required
```

Reject fields whose justification is only convenience, naming preference, serialization convenience, or avoiding a dependency that should be composed.

Prefer:

```text
fewer new types + stronger exact refs
```

over a broader local ontology.

## Ordered execution boundary

```text
MAT recovery / canonical shared-owner readiness
        ↓
#5768 exact ownership audit
        ↓
#5767 BIO-CEM deterministic fixture contract
        ↓
#5755 minimal reference envelope
        ↓
#5756 benchmark partition policy/receipt
        ↓
#5759 BIO-CEM projection
        ↓
#5757 attribution relation as needed
        ↓
#5760 ROT-EQUIP projection
        ↓
prove two-domain generality
        ↓
#5758/#5761 portfolio + public constraint evidence
        ↓
#5762–#5766 only as thin projections/adapters where existing layers do not already suffice
```

Do not open common Rust implementation before the repaired MAT interfaces required by BIO-CEM are available and the #5768 field/type audit is satisfied against current source interfaces.

## Qualification expectations

The first executable tranche should eventually prove at least:

1. missing evidence cannot increase a claim silently;
2. contradictory evidence survives aggregation;
3. dependency/common-mode lineage is not confused with row identity;
4. benchmark receipts replay deterministically;
5. performance observations cannot automatically support a mechanism;
6. HDC/model similarity cannot create causal equivalence;
7. BIO-CEM and ROT-EQUIP use the same common relations without semantic distortion;
8. policy changes do not rewrite scientific evidence;
9. recommendation/authorization/execution/outcome remain distinct;
10. scale and context transfer are never implicit.

## Non-goals

- wet-lab recipes;
- autonomous manufacturing or maintenance instructions;
- autonomous safety-critical control;
- procurement/funding authority;
- regulatory certification;
- replacing domain standards;
- creating a universal innovation/stagnation score;
- inventing duplicate identity, quantity, measurement, lineage, hypothesis, provenance, lifecycle, currentness or authority systems.

## Acceptance boundary

This document freezes architecture only. It does not claim compile, test, benchmark, experimental, manufacturing, economic, safety, regulatory or field qualification.