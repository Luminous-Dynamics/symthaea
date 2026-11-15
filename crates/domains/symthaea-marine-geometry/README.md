# symthaea-marine-geometry

Deterministic candidate manifests and a deliberately narrow Rust geometry-admission adapter for the MARINE-GEOMETRY-MANIFOLD-001 programme (#12302).

## Current scope: G0/G1 slice

This first implementation accepts a versioned JSON triangle mesh and checks:

- schema, candidate identity, declared metres and explicit right-handed frame;
- BLAKE3 digest of normalized geometry data;
- finite coordinates, face indices, duplicate and degenerate triangles;
- edge manifoldness, shared-edge winding consistency and optional closure;
- positive signed volume and a profile minimum when a closed surface is required;
- explicit UNKNOWN results for additional required checks not implemented here.

It emits deterministic JSON with typed dispositions: admissible-for-profile, invalid, incomplete, incomparable, or blocked.

## Important claim boundary

Admissible-for-profile means only that the supplied triangle mesh passed the selected triangle-mesh-integrity-v1 profile. This is not full hull admission.

This crate does not currently perform native CAD/B-rep validity, triangle-triangle self-intersection detection, surface fairness/continuity, hydrostatics, appendage/propulsor clearance, structural analysis, cavitation, CFD, class-rule checks, or manufacturing qualification. Such checks must be listed in additional_required_checks; they remain UNKNOWN and force an incomplete disposition. No automatic unit conversion, axis transform, normal repair, mesh healing, or geometry deformation is performed.

A digest is an identity/integrity aid, not a signature, publisher authentication, rights statement, or proof of engineering correctness.

## Build and test

From the workspace root:

    cargo test -p symthaea-marine-geometry
    cargo clippy -p symthaea-marine-geometry --all-targets -- -D warnings

## Candidate input

A candidate contains schema_version (currently 1), a non-empty candidate_id, geometry_digest_blake3 (initially empty while computing the digest), geometry (units, coordinate_frame, vertices, faces), and profile (profile ID, closure requirement, minimum triangle area, minimum enclosed volume, and unsupported required checks).

Compute a digest, then insert the returned value into geometry_digest_blake3:

    cargo run -p symthaea-marine-geometry --bin marine-geometry-admit -- --digest-only candidate.json

The digest is over the normalized geometry struct serialization, not the original JSON whitespace. Then run admission:

    cargo run -p symthaea-marine-geometry --bin marine-geometry-admit -- candidate.json --report admission.json

Exit codes: 0 = admissible for the declared mesh profile; 1 = a typed non-admissible disposition; 2 = CLI, I/O, or input parse error.

## Next adapters

1. Independently qualified self-intersection and mesh-quality backend.
2. Neutral CAD/B-rep adapter with separate native-solid and tessellation provenance.
3. Section-based parametric hull generator and reproducible design-space fixtures.
4. Hydrostatic calculations with explicit loading, density, trim and waterplane semantics.
5. Process-specific DFM profiles and exact hand-offs to Symtropy #2048.
6. Exact artifact/evidence receipts for #11513, #11442, Mycelix #5968 and Sol-Atlas #809.

No CFD, manufacturing or physical-test claim is made by this G0/G1 implementation.
