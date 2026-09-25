# MFG-FOUNDATION-001 — Physical Technology Improvement Engine

Status: draft research architecture
Base subject: `458c7b98d81c64b9361e252f85ef9d45132e6682`

## Purpose

Define a reusable, evidence-bounded process for identifying and improving mature physical technologies whose progress is constrained less by missing fundamental physics than by fragmented evidence, poor lifecycle feedback, weak interoperability, expensive experimentation, maintenance burden, over-customization, or design-to-manufacturing disconnects.

This is a cross-domain campaign/orchestration layer. It is **not** a second material ontology, evidence-authority system, lifecycle ledger, or autonomous invention authority.

It does not authorize manufacturing, deployment, biological experimentation, procurement, safety-critical control, or regulatory claims.

## Existing ownership — reuse first

MFG-FOUNDATION must compose existing qualified semantics rather than replacing them:

- `MAT-007` / #4336 — reproducible physical material subject identity;
- `MAT-008` / #4339 — conditioned material-property evidence, uncertainty, method class, units and source/result artifacts;
- `MAT-009` / #4340 — campaign identity, hard constraints, budgets, independent objectives and Pareto semantics;
- `MAT-010` / #4341 — failed/null/duplicate/OOD search memory;
- `MAT-013` / #4344 — provenance-complete physical sample lineage and scale state;
- `MAT-014` / #4345 — human-authorized active materials discovery orchestration;
- `CEM-001` / #4346 — low-carbon cement campaign;
- `BIO-CEM-001` / #5751 and `BIO-CEM-001A` / #5753 — living/self-healing cement campaign and literature benchmark;
- FIELD/QIF — calibrated physical observation / inspection authority where applicable;
- MFG-PROC / MFG-LIFE — manufacturing process and lifecycle semantics;
- Mycelix `BIO-MFG-001` / #3113 — operational biological lot, containment-reference and lifecycle provenance.

If one of these layers already carries identity, quantity, uncertainty, evidence authority, provenance, scale, Pareto objectives or human authorization, MFG-FOUNDATION stores a typed reference to it rather than inventing a competing representation.

## Core separation

```text
technology opportunity
!= validated intervention
!= experimentally established mechanism
!= manufacturable product
!= qualified production process
!= safe field deployment
```

The engine must preserve these distinctions structurally.

## No `stagnant=true`

"Stagnation" is not a primitive truth value.

A technology-opportunity claim must instead be grounded in concrete evidence such as:

- unusually low productivity growth;
- large conversion or distribution losses;
- high maintenance / failure burden;
- excessive specification or product-family fragmentation;
- weak interoperability;
- large deployment gap despite technically available capability;
- persistent fouling / corrosion / degradation;
- manual translation between design, controls and field execution;
- weak lab-to-field transfer;
- poor repairability or remanufacturing;
- missing lifecycle feedback.

This prevents rhetoric such as "industry X is stagnant" from entering the evidence layer as fact.

## Relationship to BIO-CEM

BIO-CEM is the first candidate campaign to exercise the causal-materials research pattern:

```text
MAT-007 material subject
+ MAT-008 conditioned observations
+ study/source lineage
+ control/intervention graph
+ mechanism attribution
+ degradation/recovery history
+ MAT-013 physical lineage when physical work begins
+ Mycelix operational provenance
```

The legacy `symthaea-materials::MaterialProperty` remains a compact bulk-engineering descriptor used by the current HDC materials functionality. BIO-CEM literature observations, viability state, study lineage and mechanism attribution must not be folded into that descriptor.

Where the MAT stack supersedes legacy descriptors for scientific authority, the legacy HDC representation should be treated as retrieval/representation infrastructure rather than the canonical evidence identity.

## Opportunity evidence contract

A cross-domain opportunity observation should reference existing domain evidence and add only what is not already owned elsewhere:

```text
TechnologyConstraintObservationV1 {
  technology_subject_ref,
  constraint_class,
  evidence_ref,
  source_lineage_ref,
  population_or_operating_context_ref,
  observation_period,
  applicability_profile_ref,
  contradiction_refs[],
  freshness_or_transfer_state
}
```

This type should not duplicate MAT-008 quantities or FIELD/QIF measurements. `evidence_ref` points to the authoritative observation where one exists.

Missing evidence is represented as missing evidence, not a neutral or favorable score.

## Candidate constraint classes

- EnergyLoss
- MaterialWaste
- FailureOrDegradationBurden
- SkilledManualTranslationBurden
- InteroperabilityFragmentation
- RepairabilityConstraint
- FieldFeedbackGap
- ExcessiveVariantFragmentation
- MaintenanceOrDowntimeBurden
- UnresolvedMechanism
- LabToFieldTransferGap
- StandardizationOpportunity
- ClosedLoopControlOpportunity
- ExperimentAccessibilityConstraint
- ManufacturingAccessibilityConstraint
- SafetyOrRegulatoryBurden

The taxonomy is descriptive. It does not imply that every recorded constraint is practically or economically solvable.

## Evidence semantics

Every opportunity claim must preserve:

- source lineage;
- observation time or study period where available;
- target population / geography / deployment context;
- measured quantity or qualitative-claim class;
- uncertainty representation;
- applicability constraints;
- contradiction references;
- evidence freshness / transfer validity.

Derived opportunity assessments may summarize evidence but may not erase conflicting observations.

## Candidate research state

Do not make one universal monotonic readiness enum span every domain. Keep independent evidence/authority planes and expose a derived research view such as:

```text
constraint observed
-> evidence qualified
-> mechanism hypotheses enumerated
-> discriminating intervention proposed
-> simulation/model evidence collected
-> experiment proposed
-> experiment observed
-> independent replication evidence collected
-> manufacturing-transfer evidence collected
-> field evidence collected
```

Later contradictory evidence may challenge earlier derived views without rewriting historical observations.

No state implies regulatory approval, safety certification, commercial readiness or causal proof unless separately evidenced.

## Mechanism-attribution graph

Observed effects may reference multiple candidate mechanisms with explicit evidence-bearing relationships.

Recommended derived states:

- SupportedUnderProfile
- ChallengedUnderProfile
- Unresolved
- NotApplicable

Prefer profile-relative names over an unqualified `Supported` where evidence applies only under a bounded condition set.

Attribution requires linked evidence. Similarity, prediction or model fit alone cannot establish causal support.

```text
HDC similarity
!= shared mechanism

model prediction
!= experimental observation

correlation
!= intervention effect

before/after improvement
!= causal attribution
```

## Portfolio selection

Do not introduce another universal weighted innovation score. Reuse the discipline of MAT-009 where possible and generalize it across technology campaigns.

Preserve an inspectable multi-objective frontier across dimensions such as:

- expected human/community utility;
- physical plausibility;
- measurable improvement potential;
- experiment affordability;
- manufacturing accessibility;
- cross-domain reuse;
- time-to-evidence;
- safety / regulatory burden;
- repairability / lifecycle leverage;
- evidence completeness / transfer risk.

Any scalar prioritization is an explicit policy layer with inspectable weights/version and must retain the underlying vector and Pareto frontier.

## Initial domain probes

Use probes that differ enough to test generality:

1. `BIO-CEM` — adaptive/living cementitious materials;
2. rotating equipment systems — motor + drive + pump/fan/compressor + piping/ducting + controls;
3. industrial thermal systems — heat recovery + heat pumps + thermal storage;
4. building controls — interoperability, commissioning and verified control transfer;
5. corrosion / protective coatings;
6. water-treatment membranes and fouling;
7. distribution transformer family/specification rationalization.

These are research probes, not product commitments.

## Why these probes are evidence-worthy

The initial external evidence scan suggests several concrete system constraints worth representing rather than asserting generic stagnation:

- global construction productivity increased only about 0.4% annually from 2000–2022 while manufacturing increased about 3% annually over the same period;
- DOE describes building-control deployment as constrained by proprietary hardware/software, fragmented data and poor interoperability, while high-performance controls can substantially reduce commercial HVAC energy use;
- DOE industrial guidance continues to treat pumps, fans, motors, process heat and compressed air as whole-system optimization opportunities; compressed air can lose more than 80% of input energy as heat;
- IEA reports commercially available heat pumps could technically supply about 20% of global industrial heat demand, while deployment remains low and integration/economics are significant constraints;
- DOE reports more than 80,000 distribution-transformer varieties in the United States and identifies inconsistent specifications as one contributor to long production times;
- corrosion remains a broad lifecycle burden across infrastructure;
- current RO literature continues to identify fouling, scaling, cleaning burden, scale-up, stability and field-transfer issues.

These examples justify an evidence registry. They do not establish that Symthaea can solve each constraint or that any candidate is commercially superior.

## Field-feedback loop

A useful cross-domain digital thread is:

```text
baseline evidence
-> hypothesis
-> proposed intervention
-> explicit authorization / execution receipt
-> post-intervention observation
-> context / confounder evidence
-> mechanism-attribution update
-> lifecycle / maintenance result
-> next campaign state
```

Required distinctions:

```text
recommendation != execution
execution != successful outcome
predicted savings != measured savings
maintenance completion != restored performance
manufacturing conformity != field durability
```

Mycelix should own operational lineage and receipts where those fit its existing semantics; Symthaea should reference them rather than duplicating the ledger.

## Qualification tests

The first implementation tranche should prove:

1. missing evidence cannot increase confidence;
2. contradictory evidence survives aggregation;
3. the same source/study lineage cannot leak across benchmark partitions;
4. performance observations do not create mechanism support automatically;
5. HDC similarity does not create causal equivalence;
6. portfolio ordering is reconstructable from evidence plus explicit policy;
7. weight sensitivity is exposed when candidate ordering is unstable;
8. domain-specific fields extend common contracts without duplicate universal material/quantity identities;
9. recommendation, authorization, execution and outcome remain distinct;
10. null/failed/negative results remain first-class evidence;
11. evidence freshness and cross-context transfer are explicit;
12. scale transfer (coupon -> lab batch -> device -> pilot -> industrial/field) is never implicit.

## Ordered implementation subjects

### Foundation

- #5755 `MFG-FOUNDATION-001A` — typed opportunity-evidence contract;
- #5756 `MFG-FOUNDATION-001B` — source-lineage-safe benchmark partitioning;
- #5757 `MFG-FOUNDATION-001C` — mechanism-attribution graph;
- #5758 `MFG-FOUNDATION-001D` — transparent multi-objective portfolio frontier.

### Generalization

- #5759 `MFG-FOUNDATION-001E` — BIO-CEM adapter;
- #5760 `MFG-FOUNDATION-001F` — second-domain industrial-equipment adapter;
- #5761 `MFG-FOUNDATION-001G` — technology-constraint evidence registry.

### Closed-loop / transfer integrity

- #5762 `MFG-FOUNDATION-001H` — field-feedback/intervention receipt boundary;
- #5763 `MFG-FOUNDATION-001I` — experiment-to-manufacturing transfer boundary;
- #5764 `MFG-FOUNDATION-001J` — negative-results/replication ledger integration;
- #5765 `MFG-FOUNDATION-001K` — evidence freshness and transfer-validity semantics;
- #5766 `MFG-FOUNDATION-001L` — human-review / authority envelope.

Implementation should merge or collapse any child whose semantics are already fully supplied by MAT/MFG/FIELD/Mycelix after exact code-location audit. Issue existence is not justification for duplicate code.

## Preferred execution order

```text
MAT/CEM/MFG reuse audit
        ↓
001A typed cross-domain refs only
        ↓
001B lineage partitioning
        ↓
BIO-CEM-001A / 001E adapter
        ↓
001C mechanism attribution
        ↓
001F rotating-equipment adapter
        ↓
prove cross-domain generality
        ↓
001D/G portfolio + external constraint evidence
        ↓
H/I/J/K/L only where existing layers do not already satisfy them
```

This deliberately postpones generic scoring and closed-loop expansion until two substantially different domains prove the common abstraction.

## Non-goals

- wet-lab recipes;
- autonomous manufacturing instructions;
- autonomous safety-critical control;
- procurement/funding authority;
- regulatory certification;
- replacing domain standards;
- inventing a second material, quantity, uncertainty or provenance identity system;
- using a single `innovation_score` or `stagnant=true` flag as truth.

## Acceptance boundary

This document freezes architecture only. It does not claim compile, test, benchmark, experimental, manufacturing, economic, safety, regulatory or field qualification.
