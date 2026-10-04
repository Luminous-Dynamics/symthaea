# Passive Function Design Architecture

## Purpose

Symthaea already has the ingredients for simulation-driven design: CSG geometry, mesh validation, structural fitness, material models, physics domains, simulation backends, and a fabrication kernel.

This document adds a design objective that cuts across those systems:

> Search for useful physical behaviour that does not require mechanically articulated solid parts or active control hardware.

The goal is not to ban motion in the physical world. Fluids may flow, fields may oscillate, materials may deform, and phases may change. The intended invariant is that the device does not require moving mechanical interfaces or commanded actuators to produce its primary function.

## Why this matters

Recent engineering work shows that topology optimization and computational metamaterials can search very large structural spaces and discover functions that are difficult to reach through conventional intuition. Shape-morphing metamaterials similarly demonstrate that geometry and nonlinear material response can produce sophisticated behaviour without conventional mechanisms.

For Symthaea, the useful shift is from:

    choose a mechanism -> parameterize it -> optimize parameters

toward:

    specify desired physical function
              |
              v
    constrain forbidden mechanisms
              |
              v
    generate topology / material hypotheses
              |
              v
    simulate and falsify
              |
              v
    retain evidence-backed candidates

This is a better fit for an architecture that already represents topology, cross-domain relationships, uncertainty, and provenance.

## Contract model

PassiveFunctionContract is the machine-readable boundary.

It specifies:

- input stimulus domain
- desired output domain
- primary physical strategy
- no-moving-parts policy

PassiveDesignEvidence is the observation side of the boundary.

It records:

- moving solid components
- mechanical joints
- active power
- commanded actuators
- external control dependency
- fluid motion
- distributed deformation
- phase change

A candidate is passively compliant only when its observed evidence satisfies the contract.

The continuous passivity score is an optimization signal, not a certification claim.

## Important semantic boundary

"No moving parts" must not be implemented as "nothing changes."

That would incorrectly exclude:

- fluidic diodes and rectifiers
- heat pipes and phase-change thermal devices
- acoustic and elastic resonators
- monolithic compliant structures
- passive electromagnetic structures
- many metamaterials whose useful response is geometric or constitutive

The architecture therefore distinguishes:

1. mechanically articulated motion — forbidden by the strict policy
2. active actuation/control — forbidden by the strict policy
3. distributed physical response — permitted
4. transport of matter or energy through fixed structure — permitted
5. phase transitions — permitted

## Search architecture

The next layer should make passive function one objective inside multi-objective generative search.

A candidate should be evaluated across at least:

    passive compliance
    structural safety
    mass / material intensity
    energy requirement
    manufacturing feasibility
    durability / fatigue
    uncertainty
    functional performance

The important design rule is that passivity is not a post-hoc filter.

It should influence candidate generation so that the search can discover mechanism-free solutions rather than generating articulated mechanisms and throwing them away.

## Physics backends

The fabrication kernel currently has a generic simulation boundary, while the broader workspace has adapters planned or present for structural mechanics, CFD, circuits, and other domains.

A high-value implementation sequence is:

1. analytical reduced-order models for fast rejection
2. deterministic geometry/topology generation
3. CFD / FEA / EM / thermal confirmation
4. uncertainty and sensitivity analysis
5. independent replay of the evidence
6. fabrication qualification

Physics-informed surrogate or operator models can accelerate stages 1–3 once the simulation corpus is large enough, but surrogate output must remain distinguishable from direct solver evidence.

## HDC role

HDC should be used primarily for:

- representing design intent
- retrieving analogous physical structures
- binding function + topology + material concepts
- maintaining diversity in candidate search
- encoding compact provenance-linked descriptors

HDC should not be used as a substitute for precision physics.

A useful architecture is:

    intent HV
       |
       +---- function/domain relations
       |
       +---- topology family
       |
       +---- material family
       |
       v
    candidate generator
       |
       v
    exact/reduced-order physics
       |
       v
    evidence capsule

This keeps HDC in its strength zone: compositional representation and search, while deterministic physics remains the truth boundary.

## Mycelix role

A successful design should not merely produce a mesh.

It should produce an evidence object describing:

- the exact design artifact
- design-contract version
- solver/backend versions
- numerical tolerances
- input assumptions
- simulation results
- uncertainty bounds
- fabrication constraints
- known limitations
- independent verification status
- lineage to prior candidates

Mycelix is a natural persistence and coordination layer for this evidence because the useful unit is not "AI says this is good"; it is "this candidate has an inspectable chain of claims and evidence."

## Anti-hallucination boundary

The system must explicitly distinguish:

Generated
The system produced the candidate.

Simulated
A specified model produced a result.

Verified
An independent or otherwise qualified verification procedure reproduced the relevant result.

Fabricated
A physical artifact was manufactured.

Measured
A physical measurement was obtained.

Validated
The measured result satisfies the declared acceptance criteria.

These states must never be collapsed into a single confidence number.

## First benchmark family

The cleanest initial benchmark is not an exotic device.

Use deliberately small reference problems where the absence of mechanisms is easy to inspect:

1. fluidic flow rectification with fixed geometry
2. passive thermal regulation using material/geometry response
3. acoustic band-pass / stop-band structure
4. passive load-distribution lattice
5. passive electromagnetic field-shaping geometry

For each problem, freeze:

- functional target
- boundary conditions
- material set
- manufacturing envelope
- passivity policy
- solver configuration

Then compare:

- conventional parameter optimization
- topology / generative search
- HDC-guided search

The principal outcome is not "AI wins."

It is whether the HDC-guided system increases useful design diversity under hard physical constraints without weakening the evidence boundary.

## Safety and engineering boundary

A passive candidate can still be dangerous.

Removing motors and controllers does not eliminate:

- pressure hazards
- stored elastic energy
- thermal hazards
- high voltage
- radiation
- reactive chemistry
- structural collapse
- fatigue failure

Therefore "passive" must never be treated as equivalent to "safe."

The safety pipeline remains authoritative.

## Current implementation status

The fabrication kernel now exposes:

- PassiveInput
- PassiveOutput
- PassiveMechanism
- NoMovingPartsPolicy
- PassiveDesignEvidence
- PassiveFunctionContract
- PassiveValidationReport
- PassiveViolation

The initial implementation is deterministic and dependency-free.

It intentionally does not claim to infer moving parts from arbitrary CAD. That requires an evidence extractor connected to geometry, assembly semantics, kinematics, and/or simulation traces.

## Next highest-value implementation

The next concrete step is to add a PassiveEvidenceExtractor that consumes fabrication artifacts and simulation metadata and emits PassiveDesignEvidence.

That extractor should start with high-confidence evidence:

- explicit assembly joints
- kinematic bodies
- actuator declarations
- machine control channels
- active electrical power budgets

Then later add inferred evidence from:

- contact topology
- relative-body motion in simulation
- fluid transport
- phase-change state transitions

This keeps the system honest while allowing progressive automation.
