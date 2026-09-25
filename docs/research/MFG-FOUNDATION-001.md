# MFG-FOUNDATION-001 — Physical Technology Improvement Engine

Status: draft research architecture
Base subject: `458c7b98d81c64b9361e252f85ef9d45132e6682`

## Purpose

Define a reusable, evidence-bounded process for identifying and improving mature physical technologies whose progress is constrained less by missing fundamental physics than by fragmented evidence, poor lifecycle feedback, weak interoperability, expensive experimentation, maintenance burden, over-customization, or design-to-manufacturing disconnects.

This is a cross-domain research/orchestration **composition**, not a new universal manufacturing ontology. It is deliberately not a second materials ontology, measurement system, evidence-dependency graph, hypothesis ontology, bottleneck taxonomy, research-portfolio allocator, manufacturing ledger, lifecycle graph, or authority system.

It does not authorize manufacturing, deployment, biological experimentation, maintenance, procurement, safety-critical control, funding, or regulatory claims.

## Governing rule — reuse before abstraction

Every proposed common field or type must first answer:

```text
who already owns this fact?
```

If a canonical owner exists, MFG-FOUNDATION stores an exact typed reference or adapter rather than copying the value, identity, lineage, calibration, validity, provenance, lifecycle state, hypothesis, planning state, execution state, or authority.

The zero-duplication gate is #5768.

The baseline implementation hypothesis is now:

```text
zero new common production types
```

A new common type must prove that existing owners cannot express the relation losslessly.

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

### Technology-function, bottleneck and system-leverage planning

MAT-OPPORTUNITY-001 #5196 already owns materials-planning concepts for:

- technology/service function;
- material/component/process bottleneck hypotheses;
- competing non-material explanations and intervention classes;
- independent opportunity dimensions with explicit unknown state;
- evidence-bounded opportunity planning and value-of-information questions.

MAT-OPPORTUNITY-003 #5245 already owns system-performance questions, candidate bottleneck variables, local/global/counterfactual sensitivity, achievable headroom, co-bottlenecks, bottleneck migration, model/applicability limits and competing non-material interventions.

Therefore MFG-FOUNDATION does **not** own a generic `TechnologyConstraintClassV1` by default.

Historical MAT-OPPORTUNITY implementation PR #5212 is not qualified: exact-head qualifier #5214 / run `35526632327` passed identity/scope/planning-boundary/lock checks and then failed Rust 1.96 rustfmt on `src/opportunity.rs`; compile/tests/Clippy did not execute. #5769 owns reproducible reconstruction of that historical source.

Useful planning semantics therefore do not imply a currently consumable implementation.

### Research portfolio planning

MAT-PORTFOLIO-001 #5247 already owns evidence-bounded research portfolio semantics such as:

- multi-resource budgets;
- staged commitments;
- exploration/exploitation separation;
- option value;
- shared-infrastructure leverage;
- common-cause/correlated failure;
- meaningful portfolio diversity;
- policy-relative dominance/incomparability;
- opportunity cost;
- VOI/VOC integration;
- continuation/kill gates;
- prospective calibration.

#5758 is therefore a cross-domain adapter into existing opportunity/portfolio/decision semantics, not a new portfolio optimizer.

### Quantities, units, frames and time

SE-SEM-001 #4861 owns canonical physical quantity-kind, unit, conversion, frame, shape and clock/time-base semantics.

Do not create local physical-value standards such as:

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

FIELD-000 #3589, FIELD-001A #3633 and SE-OBS #3695 own physical-observation, source/device/configuration, calibration/traceability, uncertainty, timestamp and raw-vs-derived evidence boundaries.

MFG-FOUNDATION should point to authoritative observations rather than copying measured payloads.

### Evidence dependency, independence and replication

ENG-EVID-INDEP-001 #4918 owns domain-neutral evidence-dependency/common-mode lineage, claim-relative independence profiles, repeat-vs-replication distinctions, calibration/clock/model/implementation common modes and indeterminate-dependence semantics.

```text
multiple observations
!= multiple independent observations
```

MFG-FOUNDATION must not create a second generic source-lineage or `LeakageGroupRef` ontology.

#5756 owns only benchmark partition policy and deterministic partition receipts over canonical dependency refs.

### Hypotheses and failure causes

Hypothesis identity stays with the domain that owns the proposition.

Examples:

- MAT-HYPOTHESIS-001 #5254 — materials mechanism/process/intervention hypotheses;
- SE-VV #3697 — engineering failure-mode/effect/cause and discrepancy-investigation hypotheses;
- domain-specific physics/research subjects where a more specific owner exists.

#5757 owns only the evidence-bearing attribution relation from an observed effect to an existing hypothesis/mechanism subject, if no existing generic causal-evidence relation already suffices.

### Applicability, validity and currentness

A generic MFG `ApplicabilityProfile` is not approved.

Existing ownership/precedent includes domain study-scope profiles, SE-VV intended-use/validity envelopes, SE-SEMANTICS applicability-review consequences and ETK authority-bearing currentness/contradiction/supersession/applicability-loss semantics.

Until an exact generic-owner audit proves a gap, MFG-FOUNDATION carries only an external validity/applicability-scope reference.

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

MFG-FOUNDATION may relate those facts to evidence and hypotheses; it should not create another generic FMEA/R&M system.

## Core separations

```text
observed system/material limitation
!= dominant bottleneck established
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
!= portfolio allocation proposal
!= human decision
!= authorization
!= execution
!= successful outcome
!= qualification/certification
```

## No `stagnant=true`

"Stagnation" is not a primitive truth value.

Public or field evidence may describe concrete limitations such as energy loss, material waste, degradation, downtime, interoperability fragmentation, excessive variants, repairability limits, weak field feedback, unresolved mechanisms, lab-to-field transfer gaps or integration barriers.

Those descriptions are evidence/profile classifications, not a new canonical cross-domain constraint enum.

A strong bottleneck/system-leverage claim should resolve through MAT-OPPORTUNITY-003-style system-question/sensitivity semantics or the appropriate system-engineering owner.

```text
large observed loss
!= root cause
!= dominant bottleneck
!= best intervention
```

## Minimal provisional common kernel

After the ownership audits, the only still-plausible shared production semantics are approximately:

```text
BenchmarkPartitionPolicyV1      // #5756, if no generic benchmark owner fits
BenchmarkPartitionReceiptV1     // #5756
MechanismAttributionEdgeV1      // #5757, if no generic causal-evidence relation fits
```

Even these are an upper bound, not an implementation target.

Previously proposed common primitives removed or demoted:

```text
TechnologyConstraintClassV1
  -> removed; overlaps MAT-OPPORTUNITY/system bottleneck semantics

TechnologyConstraintObservationV1
  -> demoted; #5755 is now a provisional adapter/query relation only

LeakageGroupRefV1
  -> removed; ENG-EVID-INDEP #4918 owns dependency/independence

MechanismHypothesisV1
  -> removed; domain owners already own hypothesis identity

ApplicabilityProfileV1
  -> not approved; existing applicability/validity owners must be exhausted first
```

If further audit eliminates another type, remove it. Issue existence does not justify code.

## Opportunity-evidence binding

#5755 no longer owns a universal constraint taxonomy. It asks whether a shared binding is necessary between already-owned technology/function/bottleneck subjects and exact evidence/context.

A provisional adapter/query shape is:

```text
OpportunityEvidenceBindingV1 {
  technology_or_function_subject_ref,
  limiting_proposition_or_variable_ref,
  evidence_ref,
  evidence_dependency_ref?,
  operating_context_ref,
  observation_period_or_time_ref?,
  applicability_or_validity_scope_ref?,
  related_or_contradictory_evidence_refs[],
}
```

This is **not yet an approved persistent type**. If existing relations can express the same binding losslessly, use them instead.

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

Conceptually, only if no existing generic relation fits:

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

BIO-CEM owns only genuinely domain-specific vocabulary such as biological observation state, healing output-plane classification, control roles, literature extraction state and method/comparability state. Material identity, measurements, source dependency, bottleneck/system questions, hypotheses, physical sample lineage, operational biological lots and authority remain with canonical owners.

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

ROT-EQUIP should add only true domain vocabulary that existing system/catalog/observation/R&M owners cannot represent, such as narrowly defined driven-system topology/role semantics if an exact owner does not already exist.

## Cross-domain generality gate

The program does not expand merely because another industry is interesting.

First prove:

```text
BIO-CEM
        \
         > same integrity / evidence-binding discipline
        /
ROT-EQUIP
```

without:

- `AnyValue`;
- free-form JSON semantic escape hatches;
- duplicate quantity/unit systems;
- duplicate observation/calibration systems;
- duplicate evidence-independence graphs;
- duplicate bottleneck/opportunity taxonomies;
- duplicate hypothesis ontologies;
- duplicate component identities;
- duplicate manufacturing/lifecycle ledgers;
- duplicate portfolio engines;
- duplicate currentness or authority systems.

Only after that proof should industrial thermal systems, building controls, corrosion/coatings, membranes, transformers or other sectors become new adapters.

## Portfolio decision layer

#5758 is now explicitly an adapter into MAT-OPPORTUNITY / MAT-PORTFOLIO / MAT-009 / SE decision semantics.

Preserve:

```text
physical-domain evidence
!= opportunity criterion
!= opportunity state
!= portfolio membership
!= allocation proposal
!= selected project
!= funded/procured project
```

Missing evidence stays missing. Existing Symthaea code may reduce cost/time-to-evidence, but cannot manufacture scientific/system importance.

Portfolio common-cause/dependency analysis should reuse ENG-EVID-INDEP rather than count differently named projects as independent bets.

If the repaired/converged planning stack can consume BIO-CEM and ROT-EQUIP through ordinary refs, #5758 may add zero production types.

## External evidence corpus

#5761 is a versioned corpus/validator over existing technology/function/system-question/limiting-proposition subjects, not a registry service or constraint taxonomy.

Every public-source claim preserves exact population/context/time/source dependency and extraction claim class. Historical evidence cannot silently become a current global fact, and newer sources do not automatically defeat older evidence.

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

No quantity of confidence, simulation success, HDC similarity, provenance completeness, portfolio priority or supporting studies can manufacture missing permission.

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

Treat zero new common production types as the starting hypothesis.

## Ordered execution boundary

```text
MAT recovery / canonical shared-owner readiness
        +
#5769 repair/qualify MAT-OPPORTUNITY planning source
        ↓
MAT convergence / reusable planning-interface decision
        ↓
#5768 exact ownership audit against current interfaces
        ↓
#5767 BIO-CEM deterministic fixture contract
        ↓
#5755 prove whether any shared evidence-binding type is necessary
        ↓
#5756 partition policy/receipt only if still missing
        ↓
#5759 BIO-CEM projection
        ↓
#5757 attribution relation only if still missing
        ↓
#5760 ROT-EQUIP projection
        ↓
prove two-domain generality
        ↓
#5758/#5761 planning adapter + public evidence corpus
        ↓
#5762–#5766 only as thin projections/adapters where existing layers do not already suffice
```

Do not open common Rust implementation before the repaired interfaces required by the adapters exist and #5768 is satisfied against those exact source subjects.

## Qualification expectations

The first executable tranche should eventually prove at least:

1. missing evidence cannot increase a claim silently;
2. contradictory evidence survives aggregation;
3. dependency/common-mode lineage is not confused with row identity;
4. benchmark receipts replay deterministically if that layer remains necessary;
5. observed performance cannot automatically support a bottleneck or mechanism claim;
6. HDC/model similarity cannot create causal equivalence;
7. BIO-CEM and ROT-EQUIP use the same composition discipline without semantic distortion;
8. planning policy changes do not rewrite scientific evidence;
9. recommendation/allocation/authorization/execution/outcome remain distinct;
10. scale and context transfer are never implicit.

## Non-goals

- wet-lab recipes;
- autonomous manufacturing or maintenance instructions;
- autonomous safety-critical control;
- procurement/funding authority;
- regulatory certification;
- replacing domain standards;
- creating a universal innovation/stagnation score;
- inventing duplicate identity, quantity, measurement, lineage, bottleneck, hypothesis, portfolio, provenance, lifecycle, currentness or authority systems.

## Acceptance boundary

This document freezes architecture only. It does not claim compile, test, benchmark, experimental, manufacturing, economic, safety, regulatory or field qualification.