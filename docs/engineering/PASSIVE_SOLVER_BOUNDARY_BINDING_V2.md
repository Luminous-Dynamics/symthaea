# Passive Solver Boundary Binding v2

Version 2 makes exact candidate-mesh identity part of every verified solver boundary binding.

## Binding contract

A concrete adapter implements SolverBoundaryBindingAdapter and must:

1. receive the exact PortInterface;
2. inspect the exact TriangleMesh supplied to the adapter;
3. select the actual solver boundary entity or patch;
4. resolve that selection into the solver-specific boundary handle;
5. provide the realized boundary-patch digest.

The binding records four linked identities:

- interface identity: what boundary was intended;
- candidate geometry identity: the semantic/CAD/CSG design;
- candidate mesh identity: the exact mesh bytes presented to the adapter;
- boundary patch identity: the actual solver entities selected.

This distinguishes "same intended design" from "same concrete solver input".

## Sealed evidence construction

Verified bindings are constructed through the mesh-backed checked constructor.
The realized boundary identity cannot be independently fabricated without
computing the candidate-mesh digest.

validate_against_candidate re-checks the exact interface, semantic geometry
digest, and candidate mesh digest before a binding is accepted as matching the
current design artifact.

## Epistemic boundary

A verified binding means the adapter established the declared mapping.
It does not prove solver convergence, discretization correctness, physical
transport, manufacturing fidelity, or experimental agreement.

physical_transport_unproven therefore remains mandatory at this layer.

## Intended flow

function
→ topology
→ typed interface
→ material candidate
→ exact candidate mesh
→ realized boundary patch
→ solver binding
→ solver execution
→ numerical evidence
→ measurement
