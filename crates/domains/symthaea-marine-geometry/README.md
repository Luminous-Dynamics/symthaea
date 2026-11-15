# symthaea-marine-geometry

Deterministic candidate manifests and a deliberately narrow Rust geometry-admission adapter for the MARINE-GEOMETRY-MANIFOLD-001 programme (#12302).

## Current scope: G0/G1 slice

This first implementation accepts a versioned JSON triangle mesh and checks:

- schema, candidate identity, declared metres and explicit right-handed frame metadata;
- BLAKE3 digest of normalized geometry data;
- finite coordinates, face indices, duplicate and degenerate triangles;
- edge manifoldness, shared-edge winding consistency and optional closure;
- vertex-link cycles for closed surfaces, rejecting pinched/disconnected local shells;
- signed volume calculated independently per edge-connected closed component, preventing a positive component from masking an inward-oriented component;
- explicit UNKNOWN results for additional required checks not implemented here.

It emits deterministic JSON with typed dispositions: admissible-for-profile, invalid, incomplete, incomparable, or blocked. The report includes per-component signed volumes and the count of vertices that fail the closed-surface link condition.

## Important claim boundary

Admissible-for-profile means only that the supplied mesh passed the selected triangle-mesh-integrity-v1 profile. This profile supports closed, consistently outward-oriented positive-volume shells. Internal cavity/void boundary semantics are not modelled and must not be represented as accepted geometry under this profile.

The axis convention and handedness are declared by the submitting artifact. This adapter checks that the declarations are present and supported; it does not independently reconstruct or verify the underlying CAD coordinate transform.

This crate does not currently perform native CAD/B-rep validity, triangle-triangle self-intersection detection, surface fairness/continuity, hydrostatics, appendage/propulsor clearance, structural analysis, cavitation, CFD, class-rule checks, or manufacturing qualification. Such checks must be listed in additional_required_checks; they remain UNKNOWN and force an incomplete disposition. No automatic unit conversion, axis transform, normal repair, mesh healing, or geometry deformation is performed.

A digest is an identity/integrity aid, not a signature, publisher authentication, rights statement, or proof of engineering correctness.

The tetrahedron and adversarial component fixtures are topology and signed-volume known-answer controls, not naval-hull benchmark results.

## Build and test

From the workspace root:

    cargo test --locked -p symthaea-marine-geometry
    cargo clippy --locked -p symthaea-marine-geometry --all-targets -- -D warnings

## Candidate input

A candidate contains schema_version (currently 1), a non-empty candidate_id, geometry_digest_blake3 (initially empty while computing the digest), geometry (units, coordinate_frame, vertices, faces), and profile (profile ID, closure requirement, minimum triangle area, minimum enclosed volume, and unsupported required checks).

Compute a digest, then insert the returned value into geometry_digest_blake3:

    cargo run --locked -p symthaea-marine-geometry --bin marine-geometry-admit -- --digest-only candidate.json

The digest is over the normalized geometry struct serialization, not the original JSON whitespace. Then run admission:

    cargo run --locked -p symthaea-marine-geometry --bin marine-geometry-admit -- candidate.json --report admission.json

Exit codes: 0 = admissible for the declared mesh profile; 1 = a typed non-admissible disposition; 2 = CLI, I/O, or input parse error.

## Next adapters

1. Independently qualified self-intersection and mesh-quality backend.
2. Neutral CAD/B-rep adapter with separate native-solid and tessellation provenance.
3. Section-based parametric hull generator and reproducible design-space fixtures.
4. Hydrostatic calculations with explicit loading, density, trim and waterplane semantics.
5. Process-specific DFM profiles and exact hand-offs to Symtropy #2048.
6. Exact artifact/evidence receipts for #11513, #11442, Mycelix #5968 and Sol-Atlas #809.

No CFD, manufacturing or physical-test claim is made by this G0/G1 implementation.
