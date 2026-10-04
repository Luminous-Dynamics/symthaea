# Passive Solver Boundary Binding v3

Version 3 makes the realized boundary selection independently verifiable.

## Binding contract

A concrete adapter implements `SolverBoundaryBindingAdapter` and must:

1. receive the exact `PortInterface`;
2. inspect the exact `TriangleMesh` supplied to the adapter;
3. select the actual solver boundary entity or patch;
4. translate that entity to the portable `BoundaryEdgeKey` representation;
5. return a draft to the core binder.

The public `bind_with_adapter` orchestration function is the only path that
stamps `solver_binding_verified=true`. Direct struct construction is sealed by
an internal evidence field, and `verified` construction is private to the crate.

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
- maximum aperture-radial residual in micrometers;
- evidence level: `AdapterAttested`.

This makes a solver binding auditable without trusting an opaque solver patch name.
`AdapterAttested` means the adapter returned a mapping draft and the core binder
accepted it after the independent candidate/interface checks. It does not mean
the external solver has accepted, executed, or numerically validated the boundary.

## Complete binding sets

`validate_binding_set` additionally enforces a one-to-one interface↔binding
mapping and rejects duplicate solver-boundary identities. All bindings in a set
must reference the same semantic candidate geometry and the same exact candidate
mesh.

`digest_binding_set` produces a deterministic order-independent identity for the
validated set, committing to the common candidate identities and every
per-binding digest. Mycelix records that collection in
`passive-solver-boundary-binding-set-v1.schema.json`.

`validate_binding_set_against_candidate` is the recommended pre-dispatch gate.
It validates the set structure and then re-checks every binding against the same
semantic candidate digest and exact `TriangleMesh`, making candidate drift a
single explicit failure rather than a caller convention.

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

## Solver-side entity attestation

The neutral core now has an explicit evidence ladder:

- `AdapterAttested`: the sealed binder accepted the adapter's mapping draft after independent candidate/interface-rim checks.
- `SolverEntityAttested`: a live-capable adapter additionally returned a non-empty solver-entity fingerprint and a mapping digest cryptographically bound to the exact interface, semantic candidate, exact mesh, realized rim, and external handle.

`SolverEntityAttested` is still an adapter provenance claim. The neutral core can verify that the receipt refers to the exact binding it is promoting, but it cannot independently inspect vendor-specific solver state. A concrete OpenFOAM/Fluent/etc. adapter must only issue this stronger receipt after performing its own solver-side entity introspection.

The transition is sealed by `promote_solver_entity_attestation` and `bind_with_adapter_and_entity_attestation`; callers cannot directly construct a stronger binding by setting an evidence flag.


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
