# Passive Solver Boundary Binding v3

Version 3 makes the realized boundary selection independently verifiable.

## Binding contract

A concrete adapter implements `SolverBoundaryBindingAdapter` and must:

1. receive the exact `PortInterface`;
2. inspect the exact `TriangleMesh` supplied to the adapter;
3. select the actual solver boundary entity or patch;
4. translate that entity to the portable `BoundaryEdgeKey` representation;
5. construct the binding through the checked constructor.

`select_boundary_patch` provides a deterministic reference selector for adapters
whose solver boundary maps directly to the candidate mesh. An adapter may instead
supply the edge selection produced by its own solver-specific entity mapping.

The checked constructor then independently verifies that:

- every selected edge is an actual boundary edge of the exact candidate mesh;
- the selected edges lie on the declared interface plane;
- the selected edges match the declared circular aperture;
- the selection is complete, not merely a subset of the expected rim;
- the selection is one connected closed boundary loop;
- the resulting patch digest is derived from interface identity, exact mesh identity,
  selected edges, perimeter, and geometric residuals.

## Evidence recorded

The realized boundary identity carries:

- candidate geometry digest: semantic/CAD/CSG identity;
- candidate mesh digest: exact mesh representation presented to the adapter;
- boundary patch digest: independently derived realized-patch identity;
- boundary edge count;
- boundary perimeter in micrometers;
- maximum interface-plane residual in micrometers;
- maximum aperture-radial residual in micrometers.

This makes a solver binding auditable without trusting an opaque solver patch name.

## Complete binding sets

`validate_binding_set` additionally enforces a one-to-one interface↔binding
mapping and rejects duplicate solver-boundary identities. All bindings in a set
must reference the same semantic candidate geometry and the same exact candidate
mesh.

`digest_binding_set` produces a deterministic order-independent identity for the
validated set, committing to the common candidate identities and every
per-binding digest.

## Why this matters

Modern solver meshes bind boundary conditions to concrete boundary entities or
patches, and mesh topology must be checked when those entities are regenerated.
A stable label is therefore useful for configuration but insufficient as sole
provenance.

The v3 contract distinguishes:

The recorded boundary matching tolerance is part of the realized-boundary identity,
so the selection decision is reproducible rather than depending on an unstated
runtime default. The current connectivity default is 50 micrometers (0.05 mm).

The mesh-side selection is the interface rim on the candidate surface. It is not,
by itself, a claim that the solver's own face/patch topology is identical; the
solver-specific external handle remains a separate adapter mapping.


same intended geometry
→ same exact mesh
→ same realized boundary selection
→ same solver handle

rather than assuming those identities are interchangeable.

## Epistemic boundary

A verified binding proves only that the declared solver-boundary mapping was
consistent with the typed interface and exact candidate mesh at binding time.

It does not prove solver convergence, discretization correctness, material
correctness, physical transport, manufacturing fidelity, or experimental
agreement.

`physical_transport_unproven` therefore remains mandatory.

## Intended flow

function
→ topology
→ typed interface
→ material candidate
→ exact candidate mesh
→ independently certified boundary patch
→ solver binding
→ solver execution
→ numerical evidence
→ measurement
