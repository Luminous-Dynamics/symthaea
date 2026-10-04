# Passive Solver Boundary Binding v1

This crate is the seam between typed passive geometry and a concrete physics solver.

It intentionally does not depend on OpenFOAM, Fluent, a particular FEM package,
or another vendor API.

## Binding contract

A concrete adapter implements SolverBoundaryBindingAdapter and must:

1. receive the exact PortInterface;
2. inspect the exact candidate mesh supplied to the adapter;
3. select the actual solver boundary entity or patch;
4. resolve the selected geometry to the solver-specific boundary entity;
5. return a digest of the exact realized boundary selection.

The resulting SolverBoundaryBinding records:

- port identity
- PortInterface digest
- solver domain and stable boundary id
- adapter identity
- opaque external boundary handle
- candidate-geometry digest (semantic/CAD/CSG identity)
- candidate-mesh digest (exact mesh presented to the adapter)
- realized-boundary/patch digest
- explicit solver-binding verification state
- explicit physical-transport-unproven state

## Why the digests matter

A solver-boundary name such as "inlet" is not sufficient provenance. Mesh
regeneration, partitioning, CAD changes, or an adapter bug can cause the same
name to refer to a different surface.

Binding therefore requires the interface identity, semantic geometry identity,
exact candidate-mesh identity, and realized patch identity. This distinguishes
"same intended design" from "same concrete solver input".

This mirrors modern simulation workflows in which physics definitions are
separated from geometry/mesh identity and boundary conditions are assigned to
specific selected entities. The adapter is the authority for selecting the solver entity, while Symthaea
verifies that the recorded candidate mesh is the exact mesh supplied to the
binding operation.

## Epistemic boundary

A verified binding proves that the adapter established the requested mapping.

It does not prove:

- solver convergence
- correct discretization
- correct material properties
- physical transport
- experimental agreement
- manufacturing fidelity

Those remain separate evidence layers.

## Intended flow

functional intent
→ typed PortInterface
→ material candidate
→ realized boundary patch
→ SolverBoundaryBinding
→ solver execution
→ numerical evidence
→ physical measurement
