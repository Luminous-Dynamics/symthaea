# Engineering Trust Kernel v1

**Status:** architecture contract / pre-implementation RFC  
**Scope:** Symthaea engineering authority, evidence, currentness, and lifecycle lineage  
**Date:** 2026-09-09

## 1. Purpose

Symthaea must not become another monolithic CAD/CAE suite. Its engineering role is to make engineering semantics, evidence, applicability, authority, and lifecycle lineage explicit across existing tools.

The Engineering Trust Kernel (ETK) defines the minimal contracts required to answer a stricter question than “does the model say this design works?”:

> Why are we currently justified in believing that this exact engineered artifact, in this exact configuration and operating domain, satisfies this exact requirement?

The kernel is deliberately epistemic and authority-constraining. It does not create physical execution authority.

## 2. Core authority theorem

These states are distinct and MUST NOT be collapsed:

```text
AI proposal
    != accepted requirement

simulation converged
    != obligation satisfied

obligation satisfied
    != independently verified

verified
    != qualified

qualified
    != approved

approved
    != currently applicable

historically applicable
    != applicable to the current revision

engineering evidence
    != operational authority
```

No lower state may reconstruct a higher-authority capability from serialization, labels, booleans, or caller-controlled metadata.

## 3. Four connected graphs

### 3.1 Engineering Semantic Graph

Describes what the engineered system means.

Representative nodes:

- Requirement
- Assumption
- Constraint
- Quantity
- Material
- Geometry
- Interface
- Behavior
- Hazard
- FailureMode
- Model
- Configuration
- ValidityDomain

### 3.2 Evidence Graph

Describes why a proposition is believed.

Representative nodes:

- Claim
- Observation
- Measurement
- Calculation
- Simulation
- Test
- Proof
- Inspection
- Calibration
- Counterexample
- Defeater
- Contradiction
- Uncertainty

Evidence objects do not own approval authority.

### 3.3 Authority Graph

Describes who or what is permitted to move an engineering statement between authority states.

Representative nodes:

- Proposal
- Verification
- Qualification
- Review
- Approval
- Certification
- Delegation
- Revocation
- Currentness

Authority is verifier-owned and scope-bound. A claim MUST NOT self-certify.

### 3.4 Asset-Lineage Graph

Describes which physical or digital artifact the engineering knowledge refers to.

Representative nodes:

- Design
- Revision
- BOM
- MaterialBatch
- ManufacturingProcess
- Machine
- Inspection
- SerialInstance
- Installation
- Maintenance
- Repair
- OperationalHistory
- Retirement

Mycelix is expected to become especially important for authority and distributed asset lineage; Xenia can bind integrity/authentication across all four graphs.

## 4. Requirement authority boundary

Generated language is proposal material only.

The required transition is:

```text
RequirementProposal
        |
        v
RequirementReview
        |
        v
AcceptedRequirement
```

An LLM, heuristic, optimizer, imported document, or external adapter may propose a requirement. None may mint an accepted engineering requirement without an explicit review/acceptance transition.

The first implementation may preserve `EngineeringRequirement` as the accepted representation for compatibility, but proposal APIs MUST return a distinct proposal type and require an explicit conversion/acceptance operation.

## 5. Simulation evidence boundary

A converged simulation is not generic proof.

A simulation may discharge an obligation only when all required bindings are exact and validated:

```text
qualified simulation result
+ exact obligation
+ exact subject / configuration baseline
+ exact solver-relevant contexts
+ required metric
+ acceptance predicate
+ validity domain
+ uncertainty / completeness requirements
= eligible obligation discharge
```

A CFD cooling result MUST NOT discharge a structural-fatigue obligation merely because both are `EvidenceKind::Simulation`.

The strict typed-context lineage developed by the Physical Agency stack should be reused rather than duplicated.

## 6. Calibration boundary

Operational or fabrication observations are evidence, not permission to silently rewrite a qualified model.

Required transition:

```text
FieldObservation
      |
      v
CalibrationProposal
      |
      v
ModelComparison
      |
      v
ValidationEvidence
      |
      v
CalibrationQualification
      |
      v
NewModelRevision
```

The previous model remains immutable, reproducible history.

## 7. Validity domains

Engineering statements are rarely universally true. Evidence and claims MUST be able to carry explicit applicability bounds, including as appropriate:

- environment
- operating envelope
- material state
- temporal scope
- spatial or geometric scale
- assumptions
- configuration baseline
- calibration state

Using evidence outside its validity domain yields `NeedsReverification`, `Invalidated`, or `Unknown`; it MUST NOT silently remain current.

## 8. Semantic staleness

ETK treats engineering change as semantic impact, not merely file modification.

Minimum applicability states:

```rust
pub enum ApplicabilityState {
    Current,
    HistoricallyValid,
    NeedsReverification,
    Invalidated,
    Superseded,
    Unknown,
}
```

When a dependency changes, the kernel computes an impact frontier over claims, evidence, verification, qualification, and approval.

Invalidated evidence is never deleted. It remains historically valid for the baseline it actually supported.

## 9. Continuity receipts

Blind invalidation is safe but expensive; blind reuse is unsafe.

ETK should eventually support an `EvidenceContinuityReceipt` proving that evidence remains applicable across a semantic delta.

A continuity proof MUST establish both:

1. known dependencies were unchanged or remain valid; and
2. the new revision did not introduce a new dependency-generating surface relevant to the evidence.

This produces a scalable middle path:

```text
invalidate everything -> expensive
reuse everything      -> dangerous
prove continuity       -> scalable
```

## 10. Evidence independence

Evidence count is not independent-evidence count.

Corroborating analyses may share hidden common dependencies such as:

- the same material database
- the same calibration source
- the same geometry reconstruction
- the same boundary-condition assumption
- the same solver defect
- the same sensor calibration

The evidence graph should preserve these dependencies so Symthaea can report correlated support rather than falsely multiplying confidence.

## 11. Qualification coverage

Qualification MUST be coverage-complete over explicitly required scenario cells. Easy cases may not average away absent hard cases.

Future reusable primitives:

- `QualificationMatrix`
- `ScenarioCell`
- `CoverageRequirement`
- `CoverageReceipt`

Qualification succeeds because required coverage is satisfied, not because an aggregate score exceeds a threshold.

## 12. Uncertainty and current permitted envelope

ETK MUST preserve orthogonal epistemic state rather than collapse engineering trust into a master confidence scalar.

At minimum, downstream reasoning should be able to distinguish:

- evidence quality
- coverage
- uncertainty
- model applicability
- reproducibility
- independence
- verification status
- qualification status
- authority status
- currentness
- contradiction state

Current permitted operation should be derived from the intersection of qualified design, as-built, operational, environmental, and degradation envelopes.

New observation may preserve or tighten that envelope. Expansion requires explicit new qualification evidence.

> Uncertainty cannot spontaneously create capability.

## 13. Lifecycle twins

The generic phrase “digital twin” is insufficient for authority reasoning. ETK distinguishes:

- **Design Twin** — what is intended to exist
- **As-Built Twin** — what manufacturing and inspection evidence says actually exists
- **Operational Twin** — what telemetry says the physical instance is currently doing

The system should reason over divergence among all three without conflating them.

## 14. External interoperability boundaries

ETK should wrap established engineering standards rather than invent a proprietary universal modeling language.

Target boundaries include:

- SysML v2 / KerML — systems and model semantics
- OSLC RM — requirements lifecycle linking
- STEP AP242 — product definition / MBD
- FMI 3.x — model exchange and co-simulation
- AutomationML / IEC 62714 — plant and automation engineering exchange
- QIF / ISO 23952 — inspection and metrology
- OPC UA / Asset Administration Shell — operational asset representation
- SACM — structured assurance cases

Adapters translate into ETK semantics and evidence; they do not gain authority merely by importing data.

## 15. Explicit non-goals

ETK v1 does NOT:

- replace CAD, CAE, PLM, MES, QMS, SysML, or solver ecosystems
- implement a new universal units engine
- duplicate Symthaea causal, counterfactual, uncertainty, physics, or optimization primitives
- grant physical execution authority
- treat AI output as engineering evidence by default
- treat simulation convergence as proof
- treat serialization as authority reconstruction
- silently mutate qualified models from field observations
- delete historically valid evidence when a revision changes
- reduce trust to a single confidence number

## 16. Immediate corrective tranche

The first implementation tranche is intentionally small:

1. **ETK-1 — separate proposed and accepted requirements**
   - Broca-generated requirements become `RequirementProposal`.
   - Acceptance is an explicit operation.
   - Existing accepted `EngineeringRequirement` remains compatible initially.

2. **ETK-2 — bind simulation evidence to exact proof obligations**
   - Convergence alone cannot discharge obligations.
   - Reuse strict simulation context lineage from the Physical Agency work.

3. **ETK-3 — make calibration evidence-qualified**
   - Metrology produces calibration proposals.
   - No direct mutation of the active qualified model.

Follow-on work adds exact baseline identity, validity domains, staleness propagation, continuity receipts, reproducible execution evidence, lifecycle twin identity, manufacturing lineage, inspection ingestion, and qualification matrices.

## 17. Conformance benchmark

The first end-to-end benchmark should be a deliberately small bracket with:

- three requirements
- two assumptions
- one design baseline
- material context
- analytical calculation
- FEA result
- manufacturing route
- inspection result
- simulated strain telemetry

The benchmark then attacks the evidence chain by changing material, geometry, tolerances, solver version, mesh, calibration state, units, process qualification, and revision identity; by corrupting artifacts; by replaying old verification; and by injecting contradictory inspection evidence.

The benchmark passes only when the kernel can answer:

> Why are we currently justified in believing this exact physical bracket, as manufactured and presently operated, satisfies this exact requirement?

and separately identify every proposition it cannot establish.

## 18. Design rule

The Engineering Trust Kernel should optimize neither for maximum autonomy nor maximum paperwork. It should optimize for **explicit, reproducible, scope-correct justification**.

Symthaea reasons about what evidence would justify an engineering conclusion, what remains uncertain, what assumptions it depends on, what could falsify it, and whether that evidence remains applicable to the exact artifact under consideration.

Mycelix preserves who produced that evidence, who may see it, which artifact it refers to, who has authority to accept it, and how its lineage survives organizational boundaries.
