# symthaea-marine-geometry

Deterministic candidate manifests and a deliberately narrow Rust geometry-admission adapter for the MARINE-GEOMETRY-MANIFOLD-001 programme (#12302).

## G2a section-family generator

A separate implementation increment is under development on branch `research/marine-section-generator-g2a-20261115` for issue #12338. It accepts bounded, normalized keel-to-sheer half-section control polylines at strictly ordered x/L stations and deterministically samples physical starboard-side sections in metres. The section-family artifact carries separate BLAKE3 digests for the normalized source specification and generated sample coordinates, along with dimensions and frame metadata. The input is bounded to 512 stations, 128 controls per station, 64 subdivisions per segment, and 1,000,000 output points. It is a parametric section baseline, not a completed hull loft, watertight vessel surface, spline fairing or validated naval-architecture generator.

## G2b waterplane-capped surface loft

G2b is under development for [issue #12357](https://github.com/Luminous-Dynamics/symthaea/issues/12357) on the stacked branch `research/marine-waterplane-loft-g2b-20261115`.

It turns a versioned section-family artifact into a deterministic closed triangle mesh for the **displacement volume below the declared design waterplane**. The port side is mirrored from the starboard samples, longitudinal station strips are connected, a waterplane cap closes the top, and planar bow/stern end caps close the volume. The first implementation accepts only convex section rings with identical point counts at each station. It rejects unsupported/non-convex input rather than healing or resampling it.

The waterplane cap is not a deck surface or a freeboard model. This output is a watertight displacement-volume mesh, not a full neutral CAD hull, smooth surface, hydrostatic qualification, self-intersection certification, DFM pass or hydrodynamic result.

A separate CLI accepts a request containing `section_family` and `profile`:

    cargo run --locked -p symthaea-marine-geometry --bin marine-waterplane-loft -- loft-request.json --output /tmp/lofted-displacement.json

## G3a neutral CAD translation evidence contract

The G3a contract is under development for [issue #12376](https://github.com/Luminous-Dynamics/symthaea/issues/12376) on branch `research/marine-cad-interop-g3a-20261115`. It defines typed source/STEP digests, explicit geometry semantics, representation kind, AP242 edition, exporter/importer family/build/configuration identities, tolerance profiles, required checks, measured metrics and a deterministic receipt digest.

The initial qualification profile accepts released ISO 10303-242:2025 Edition 4; it does not silently promote AP214 or the Edition 5 committee draft. Full-external-hull semantics require explicit deck/freeboard representation and an exact B-rep target. G2b displacement-volume shells remain a distinct semantic class.

Verify a receipt from the workspace root:

    cargo run --locked -p symthaea-marine-geometry --bin marine-cad-receipt-verify -- translation-receipt.json

Exit codes: 0 = qualified within the declared profile; 1 = not qualified/incomplete; 2 = CLI, I/O or parse error.

The contract is not a STEP parser/writer or CAD kernel. It validates receipt completeness and consistency; real file conversion and independent B-rep validation belong to later G3b/G3c backend adapters.

See [CAD_INTEROP_PROTOCOL.md](./CAD_INTEROP_PROTOCOL.md) for the standard/version policy, representation semantics, validation metrics, negative controls and cross-kernel benchmark sequence.

## G3b CAD backend capability evaluation

G3b is under development for [issue #12381](https://github.com/Luminous-Dynamics/symthaea/issues/12381) on branch `research/marine-cad-kernel-bakeoff-g3b-20261115`. It adds a typed backend capability manifest and evaluator. A capability can qualify only with both a frozen fixture-manifest digest and a test-report digest; missing/unknown results remain incomplete and failed results reject the selected profile.

The current comparison candidates are OCCT 8.0.1, the experimental Rust `cad-kernel`, `brep-to-step` and `vcad_kernel_step`. Their documented STEP schema coverage differs; do not treat AP214, AP242 Edition 2 and AP242:2025 Edition 4 as equivalent.

Run a backend capability manifest against a frozen requirement profile:

    cargo run --locked -p symthaea-marine-geometry --bin marine-cad-backend-evaluate -- backend-evaluation-request.json

Exit codes: 0 = qualified within profile; 1 = incomplete; 2 = rejected; 3 = CLI, I/O, parse or manifest-integrity error.

See [CAD_BACKEND_EVALUATION.md](./CAD_BACKEND_EVALUATION.md) for candidate capabilities, CAx-IF fixture families and the cross-kernel bake-off. No backend is qualified by this manifest alone; real backend tests remain required.

## G3b Part 21 STEP header inspection

The G3b backend branch also includes a minimal ISO 10303-21 Part 21 envelope inspector. It checks the start/end markers, HEADER/DATA/ENDSEC ordering, a single FILE_SCHEMA statement and an exact match against a caller-supplied frozen list of schema identifiers. It ignores text inside comments and quoted strings.

Run it with:

    cargo run --locked -p symthaea-marine-geometry --bin marine-step-header-inspect -- model.step ap242e4-header-profile.json

Important: this inspects the Part 21 envelope and exact FILE_SCHEMA string only. The required edition is profile metadata; the header string alone does not prove ISO edition/conformance, valid DATA entities, B-rep correctness, PMI preservation or geometric equivalence.

The composed `marine-cad-translation-verify` CLI reads a real STEP file, a CAD translation receipt and a frozen header profile. It verifies that the actual STEP bytes match the receipt's file digest and that the Part 21 envelope/FILE_SCHEMA identifiers match the profile:

    cargo run --locked -p symthaea-marine-geometry --bin marine-cad-translation-verify -- model.step translation-receipt.json ap242e4-header-profile.json

This is still an envelope-and-identity check. It does not parse B-rep DATA entities or perform independent geometric/PMI validation.

## Research benchmark protocol

The matched external comparison, metrics, uncertainty policy, target ladder, and implementation sequence are specified in [BENCHMARK_PROTOCOL.md](./BENCHMARK_PROTOCOL.md). It is a frozen research protocol and target set, not a claim that the current crate generates hulls or has achieved hydrodynamic performance.

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
