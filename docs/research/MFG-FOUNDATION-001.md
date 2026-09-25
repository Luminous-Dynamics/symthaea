# MFG-FOUNDATION-001 — Physical Technology Improvement Engine

Status: draft research architecture
Base subject: `458c7b98d81c64b9361e252f85ef9d45132e6682`

## Purpose

Define a reusable, evidence-bounded process for identifying and improving mature physical technologies whose progress is constrained less by missing fundamental physics than by fragmented evidence, poor lifecycle feedback, weak interoperability, expensive experimentation, maintenance burden, or design-to-manufacturing disconnects.

This is not an autonomous invention authority and does not authorize manufacturing, deployment, biological experimentation, or safety-critical action.

## Core separation

```text
technology opportunity
!= validated intervention
!= experimentally established mechanism
!= manufacturable product
!= safe field deployment
```

The engine must preserve these distinctions structurally.

## Relationship to BIO-CEM

BIO-CEM is the first candidate campaign to exercise the causal-materials research pattern:

```text
material descriptor
+ experimental observation
+ control/intervention graph
+ mechanism attribution
+ degradation/recovery history
+ manufacturing provenance
```

The existing `symthaea-materials::MaterialProperty` remains the compact bulk-engineering descriptor. BIO-CEM observations, viability state, study lineage, and mechanism attribution belong in separate evidence structures rather than being folded into intrinsic material properties.

## Opportunity dimensions

A candidate technology may be assessed only on explicit evidence-bearing dimensions:

- energy loss or conversion inefficiency
- material waste
- failure / degradation burden
- manual skilled-labor burden
- interoperability fragmentation
- repairability constraints
- field-feedback quality
- excessive configuration / variant count
- maintenance cost and downtime
- unresolved causal mechanisms
- lab-to-field transfer weakness
- standardization potential
- adaptability / closed-loop-control potential
- experiment accessibility
- manufacturing accessibility
- safety / regulatory burden

Missing evidence is represented as missing evidence, not a neutral or favorable score.

## Evidence semantics

Every dimension must carry:

- source lineage
- observation time or study period where available
- target population / deployment context
- measured quantity or qualitative claim type
- confidence / uncertainty representation
- applicability constraints
- contradiction references

Derived opportunity assessments may summarize evidence but may not erase conflicting observations.

## Candidate state machine

```text
ObservedConstraint
    -> EvidenceQualified
    -> MechanismHypothesized
    -> InterventionProposed
    -> SimulationQualified
    -> ExperimentDesigned
    -> ExperimentObserved
    -> ReplicationQualified
    -> ManufacturingQualified
    -> FieldQualified
```

Transitions are monotonic only with respect to evidence state; later contradictory evidence may challenge or revoke prior qualification claims.

No state implies regulatory approval, safety certification, commercial readiness, or causal proof unless those are separately evidenced.

## Mechanism-attribution matrix

Each observed effect may reference one or more candidate mechanisms with an explicit attribution state:

- Supported
- Challenged
- Unresolved
- NotApplicable

Attribution requires linked evidence. Similarity, prediction, or model fit alone cannot set `Supported`.

```text
HDC similarity
!= shared mechanism

model prediction
!= experimental observation

correlation
!= intervention effect
```

## Portfolio selection

Do not reduce candidate selection to a single opaque scalar. Preserve a multi-objective frontier across at least:

- expected societal usefulness
- physical plausibility
- measurable improvement potential
- experiment affordability
- manufacturing accessibility
- reuse across domains
- time-to-evidence
- safety / regulatory burden

Any scalar prioritization must be an explicit policy layer with inspectable weights and must retain the underlying vector.

## Initial domain probes

The first bounded probes should test whether the same evidence model generalizes across substantially different mature technologies:

1. cementitious self-healing / adaptive materials
2. rotating equipment systems (pump + motor + drive + piping / ducting)
3. industrial thermal systems (heat recovery + heat pumps + thermal storage)
4. corrosion / protective coatings
5. water-treatment membranes and fouling
6. distribution transformer design / lifecycle families

These are research probes, not product commitments.

## Qualification tests

The first implementation tranche should prove:

1. missing evidence cannot increase confidence;
2. contradictory evidence survives aggregation;
3. the same source lineage cannot leak across benchmark partitions;
4. performance observations do not create mechanism support automatically;
5. HDC similarity does not create causal equivalence;
6. candidate ranking can be reconstructed from source evidence and explicit policy;
7. a high-potential but high-risk candidate cannot silently outrank a safer candidate through hidden weights;
8. domain-specific fields can extend the common observation contract without creating duplicate universal material or quantity identities.

## Non-goals

- wet-lab recipes
- autonomous manufacturing instructions
- autonomous safety-critical control
- regulatory certification
- replacing domain standards
- inventing a second material identity system
- using a single "innovation score" as truth

## Proposed follow-on subjects

- `MFG-FOUNDATION-001A`: typed opportunity-evidence contract and uncertainty semantics
- `MFG-FOUNDATION-001B`: source-lineage-safe benchmark partitioning
- `MFG-FOUNDATION-001C`: mechanism-attribution graph
- `MFG-FOUNDATION-001D`: transparent multi-objective portfolio frontier
- `MFG-FOUNDATION-001E`: BIO-CEM adapter proving the common model
- `MFG-FOUNDATION-001F`: second-domain adapter (rotating equipment or industrial thermal) proving generality

## Acceptance boundary

This document freezes architecture only. It does not claim compile, test, benchmark, experimental, manufacturing, or field qualification.
