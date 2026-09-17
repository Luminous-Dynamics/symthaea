# Symthaea Systems Engineering Architecture Profile v1

**Status:** architecture freeze candidate

## Purpose

This profile defines the systems-engineering semantic and lifecycle layer above the Engineering Trust Kernel (ETK).

It deliberately does **not** create a second assurance, evidence-admission, currentness, qualification, deployment, manufacturing, or actuation authority path.

The core architectural split is:

```text
Symthaea cognition
    -> systems-engineering semantic graph
    -> standards / analysis adapters
    -> deterministic engineering tools
    -> candidate evidence
    -> ETK evidence / currentness / assurance boundaries
    -> separately authorized downstream transition
```

## Governing theorems

```text
model relation
!= accepted engineering truth
!= admitted evidence
!= current requirement satisfaction
!= qualified design
!= deployment / manufacturing / actuation authority
```

```text
syntactically valid
!= semantically faithful
!= analytically established
!= verified
!= qualified
```

```text
HDC/LTC association
!= engineering relationship
!= causal fact
!= safety claim
```

Associative cognition may propose hypotheses or graph mutations. It may not directly mint accepted requirements, verification closure, assurance, qualification, or physical authority.

## Relationship to ETK

ETK remains the authority/evidence substrate.

The systems-engineering layer owns descriptive and lifecycle semantics such as:

- stakeholder needs;
- accepted and proposed requirements;
- functions and logical decomposition;
- physical/logical components;
- interfaces and allocations;
- assumptions and constraints;
- hazards, controls, and risk-analysis relationships;
- verification and validation intent;
- configurations, revisions, decisions, and changes;
- explicit impact relationships;
- links to model, simulation, test, telemetry, and standards artifacts.

ETK remains responsible for distinctions such as:

```text
proposal != accepted requirement
computation != admitted evidence
admitted evidence != discharge receipt
receipt != present-tense discharge
present discharge != complete requirement satisfaction
requirement satisfaction != qualified design
qualified design != downstream physical authority
```

The SE layer must consume ETK facts through typed, read-only boundaries rather than reconstructing authority from strings, labels, booleans, or serialized audit records.

## Three graph families

Symthaea engineering should maintain three distinct graph families.

### 1. Engineering semantic graph

Represents what the engineering model says exists and how engineering concepts relate.

Examples:

```text
StakeholderNeed
Requirement
Function
Component
Interface
Allocation
Constraint
Assumption
Hazard
Control
VerificationActivity
ValidationActivity
Configuration
Decision
Change
```

### 2. Evidence / authority graph

Owned by ETK and related trust/assurance crates.

Examples include admitted evidence, immutable receipts, present-currentness facts, requirement-assurance facts, trusted-time capabilities, qualification records, and transition permissions.

The SE semantic graph may reference these identities, but must not mint or reinterpret them.

### 3. Cognitive associative graph

Owned by Symthaea cognition/HDC/LTC.

This graph may encode similarity, analogy, novelty, temporal association, candidate causal structure, anomaly, or hypothesis.

A cognitive relation becomes an engineering semantic relation only through an explicit proposal/review transition. A semantic relation becomes authority-bearing only through the appropriate ETK/qualification path.

## Minimum typed semantic model

The first implementation tranche should support the following role-safe entities.

### StakeholderNeed

Captures stakeholder intent before derivation into engineering requirements.

Required semantics:

- stable local identity;
- statement;
- source/provenance reference;
- lifecycle state;
- revision identity;
- optional parent need / rationale links.

### Requirement

Represents accepted or proposed engineering intent without claiming verification.

Required semantics:

- stable identity;
- revision identity;
- statement;
- requirement class;
- criticality;
- derivation/source relationships;
- applicable configuration / subject scope;
- expected verification method(s);
- constraints and assumptions;
- explicit lifecycle state.

Existing ETK accepted-requirement semantics remain authoritative where applicable. The SE model must adapt to those semantics rather than define a competing accepted-requirement authority object.

### Function

Represents intended behavior independent of a particular component allocation.

Relationships may include:

```text
Requirement -> satisfied-by-design-intent -> Function
Function -> decomposes-to -> Function
Function -> allocated-to -> Component
```

These relationships are descriptive design semantics, not proof of requirement satisfaction.

### Component

Represents logical, software, hardware, human, organizational, or physical system elements.

Minimum properties:

- typed component class;
- revision;
- configuration membership;
- interfaces;
- allocations;
- optional external-model references.

### Interface

Interfaces are first-class engineering objects, not strings on components.

Minimum semantics:

- endpoints;
- directionality where applicable;
- transported matter / energy / information / control semantics;
- units / quantity dimensions where applicable;
- assumptions and constraints;
- interface revision.

### Constraint

Represents an explicit design or operating restriction.

Constraint existence does not establish that the constraint is currently satisfied.

### Assumption

Assumptions are first-class, versioned, impact-bearing objects.

Changing or invalidating an assumption must be able to trigger change-impact analysis over downstream requirements, analyses, hazards, and evidence applicability.

### Hazard

Represents a potential unsafe or loss-producing system condition.

Hazard identification is not equivalent to risk acceptance or safety closure.

### Control / Mitigation

Represents a proposed or accepted design control associated with one or more hazards.

Control presence does not prove control effectiveness.

### VerificationActivity

Describes the intended method for evaluating an engineering proposition.

Examples:

```text
Analysis
Simulation
Test
Inspection
Telemetry
FormalProof
StandardsEvidence
```

The activity itself is not evidence. Its outputs must still pass ETK admission where authority is required.

### Configuration

Represents an exact engineering configuration or configuration family.

Requirements, assumptions, analyses, evidence, and operational observations must be able to declare the configuration(s) to which they apply.

### Change

Represents a proposed, accepted, implemented, or observed change between semantic revisions.

A change must support explicit impact traversal rather than relying only on textual/file references.

### Decision

Captures engineering decision records, alternatives considered, rationale, assumptions, trade-space inputs, evidence references, and responsible decision authority.

A decision record is an auditable record, not self-authenticating authority.

## Required relationship classes

The first graph should support typed relationships rather than generic `edge_type: String` semantics.

Initial relationship vocabulary:

```text
derived_from
refines
allocated_to
interfaces_with
constrained_by
assumes
threatened_by
mitigated_by
verified_by_intent
validated_by_intent
depends_on
supersedes
contradicts
changes
impacts
invalidates_candidate
```

The exact Rust representation may use enums/typed edge records, but protocol semantics must remain role-safe and versioned.

## Change-impact semantics

Change impact is a primary capability, not a reporting afterthought.

Given a changed semantic object, the graph should be able to produce explicit paths such as:

```text
changed component
    -> affected interface
    -> affected function
    -> affected requirement
    -> affected verification activity
    -> candidate evidence applicability review
```

or:

```text
invalidated assumption
    -> affected analysis model
    -> affected hazard control
    -> affected requirement-verification member
    -> ETK currentness / applicability re-evaluation required
```

The SE graph itself does not invalidate ETK capabilities by mutation. It produces typed impact facts or requests that the ETK/currentness layer can evaluate through its own authority rules.

## Standards posture

Symthaea should integrate with existing standards rather than create a proprietary systems-engineering modeling language.

### SysML v2 / KerML

Primary systems-model interchange target.

Initial posture:

1. read-only import / projection;
2. deterministic semantic mapping into the internal SE graph;
3. round-trip-safe identity mapping only after read-side semantics are stable;
4. write support only after mutation/authority boundaries are explicit.

A syntactically valid SysML model remains descriptive model content, not engineering evidence.

### ReqIF

Requirements interchange target where existing toolchains use ReqIF.

ReqIF import must preserve external identifiers, revision/context metadata, provenance, and unsupported fields rather than silently dropping semantics.

### SACM

Assurance-case projection/export target.

ETK remains the underlying source of assurance/evidence/currentness truth. SACM is an interchange/projection representation, not a second assurance authority.

### RAAML

Risk-analysis projection/import target for structured hazard/risk relationships.

RAAML-derived risk models remain engineering semantic/analysis artifacts until separately qualified.

### FMI

Model-exchange / co-simulation adapter target.

FMU execution must route through existing simulation/evidence boundaries. Successful execution is never sufficient for ETK admission.

### AADL

Analyzable cyber-physical architecture target, especially for latency, deployment, resource budgets, buses, software/hardware allocation, operational modes, and fault analysis.

AADL diagnostics and analysis results remain computations until admitted through ETK.

## Cognitive role

HDC/LTC should be used where associative cognition is an advantage:

- cross-domain structural analogy;
- missing relationship proposals;
- candidate common-cause dependencies;
- anomaly clustering;
- trade-space neighborhood search;
- temporal evolution / degradation association;
- retrieval of similar prior engineering episodes;
- hypothesis generation.

HDC/LTC must not directly:

- accept requirements;
- mark requirements satisfied;
- discharge proof obligations;
- qualify models;
- authorize deployment;
- authorize manufacturing;
- authorize actuation.

The intended flow is:

```text
associative hypothesis
    -> explicit candidate semantic relation
    -> deterministic / analytical / simulation / review path
    -> candidate evidence
    -> ETK admission / currentness / assurance
```

## Model revision and identity rules

Every semantic object that can affect engineering meaning must have explicit revision semantics.

At minimum, changes to the following must not silently preserve semantic identity:

- requirement statement / class / criticality / expected verification;
- function definition;
- component revision / allocation;
- interface contract;
- units or quantity dimensions;
- assumption content or applicability;
- constraint content;
- hazard semantics;
- verification intent;
- configuration membership;
- external model reference / revision.

Audit labels and human-readable names are not authority-bearing identity.

## Non-goals

SE v1 does not:

- replace ETK;
- replace SysML, AADL, ReqIF, SACM, RAAML, FMI, CAD, CAE, PLM, or solver ecosystems;
- implement a new FEA/CFD/SPICE solver;
- grant physical execution authority;
- turn HDC similarity into engineering truth;
- collapse requirement, evidence, assurance, qualification, deployment, and actuation into one score;
- introduce a universal engineering confidence scalar;
- claim certification compliance merely because a standard-shaped model can be exported.

## Initial PR sequence

### SE-001 — typed engineering lifecycle graph

Create a small isolated crate for role-safe semantic entities, revisions, typed relationships, deterministic serialization/identity, graph validation, and change-impact traversal.

No ETK authority type is reimplemented.

Exit gate:

```text
valid graph
!= verified design
```

Tests must include dangling reference rejection, wrong-role identity rejection, deterministic ordering, duplicate-edge semantics, revision sensitivity, and explicit cycles where they are disallowed by relation type.

### SE-002 — requirement decomposition and ETK bridge

Connect the semantic requirement/decomposition graph to existing ETK requirement/obligation/assurance boundaries.

Exit gates:

```text
complete decomposition
!= current satisfaction
```

and

```text
current satisfaction can only be consumed from ETK-derived facts
```

### SE-003 — SysML v2 read-only bridge

Import/project a conservative subset of SysML v2/KerML into the semantic graph with explicit unsupported-semantics reporting.

No mutation/export authority in the first tranche.

### SE-004 — semantic model snapshot lineage

Bind exact system-model snapshots/configurations to engineering subject/twin/revision context and expose deterministic model-change sets.

### SE-005 — change-impact engine

Produce typed transitive impact paths and candidate applicability-review requests across assumptions, interfaces, functions, requirements, hazards, verification activities, and external model references.

### SE-006 — SACM assurance projection

Project ETK facts into SACM-compatible assurance structures without creating an alternate assurance path.

### SE-007 — RAAML risk projection

Add structured hazard / FMEA / FTA / STPA-oriented mappings while retaining candidate-analysis semantics.

### SE-008 — FMI adapter

Add exact FMU identity/metadata binding and isolated model execution through the existing simulation bridge.

### SE-009 — AADL analysis bridge

Add analyzable architecture projection and deterministic external-tool result capture for cyber-physical resource/timing/deployment analyses.

### SE-010 — Systems Engineering Gym v1

Freeze objective benchmark tasks for traceability, interface defects, hidden common-cause dependencies, change impact, verification completeness, stale assumptions, model/tool disagreement, uncertainty, and abstention.

The benchmark must track false-authority rate separately from ordinary task accuracy.

### SE-011 — HDC/LTC engineering hypothesis layer

Only after deterministic semantic and assurance boundaries qualify, allow cognitive layers to propose model relationships and alternatives.

### SE-012 — continuous systems engineering loop

Compose model revisions, simulation/analysis, verification, telemetry, operational observations, anomaly detection, change proposals, applicability review, and re-verification while preserving authority boundaries.

## Qualification posture

Each implementation PR should use the same discipline already established elsewhere in Symthaea:

```text
authored source
-> exact-head formatting / compile / test / lint
-> deterministic fixtures
-> independent reference oracle where identity/authority semantics warrant one
-> adversarial negative corpus
-> integration only after lower contracts qualify
```

Scientific/engineering claims must remain narrower than implementation correctness claims.

## Long-term target

The target is not an "AI engineer persona."

The target is a continuously maintained engineering epistemic system capable of representing:

```text
stakeholder intent
-> requirements
-> functions
-> architecture
-> interfaces
-> assumptions / constraints
-> hazards / controls
-> analysis / simulation / test intent
-> evidence
-> operational observations
-> configuration change
-> applicability review
-> redesign
```

while always preserving the distinction between:

```text
what Symthaea proposes
what the model states
what deterministic tools calculate
what evidence supports
what ETK currently permits us to rely upon
what authorized humans or external institutions approve
```
