# symthaea-marine-geometry

Deterministic candidate manifests and a deliberately narrow Rust geometry-admission adapter for the MARINE-GEOMETRY-MANIFOLD-001 programme (#12302).

## G2a section-family generator

A separate implementation increment is under development on branch `research/marine-section-generator-g2a-20261115` for issue #12338. It accepts bounded, normalized keel-to-sheer half-section control polylines at strictly ordered x/L stations and deterministically samples physical starboard-side sections in metres. The section-family artifact carries separate BLAKE3 digests for the normalized source specification, generated sample coordinates, and a family manifest that binds dimensions, frame metadata and longitudinal end-condition semantics. The input is bounded to 512 stations, 128 controls per station, 64 subdivisions per segment, and 1,000,000 output points. It is a parametric section baseline, not a completed hull loft, watertight vessel surface, spline fairing or validated naval-architecture generator.

## G2b waterplane-capped surface loft

G2b is under development for [issue #12357](https://github.com/Luminous-Dynamics/symthaea/issues/12357) on the stacked branch `research/marine-waterplane-loft-g2b-20261115`.

It turns a versioned section-family artifact into a deterministic closed triangle mesh for the **displacement volume below the declared design waterplane**. The port side is mirrored from the starboard samples, longitudinal station strips are connected, a waterplane cap closes the top, and planar bow/stern end caps close the volume. The first implementation accepts only convex section rings with identical point counts at each station. It rejects unsupported/non-convex input rather than healing or resampling it.

The waterplane cap is not a deck surface or a freeboard model. This output is a watertight displacement-volume mesh, not a full neutral CAD hull, smooth surface, hydrostatic qualification, self-intersection certification, DFM pass or hydrodynamic result.

Section specs with `schema_version: 1` remain valid for legacy planar/planar ends. For schema v2, the optional `longitudinal_end_conditions` object explicitly identifies bow and stern as `planar-end-cap` or `collapsed-centreline`. A runnable pointed-bow/planar-stern example is `examples/pointed-bow-planar-stern-section-spec.json`:

    cargo run --locked -p symthaea-marine-geometry --bin marine-section-generate -- examples/pointed-bow-planar-stern-section-spec.json --output /tmp/pointed-sections.json

The separate loft CLI accepts a request containing `section_family` and `profile`:

    cargo run --locked -p symthaea-marine-geometry --bin marine-waterplane-loft -- loft-request.json --output /tmp/lofted-displacement.json

Section-family artifacts are schema v2 and bind end conditions, dimensions and coordinate-frame metadata in a separate family-manifest digest. Loft artifacts are schema v2 and bind exact station/point counts, closure labels and bow/stern semantic face ranges. Old v1 section-family / loft files may still be parsed where defaults exist, but they are not admissible under the v2 verifier; regenerate them from their source specs rather than silently assuming their closure semantics.

## G2e explicit pointed bow/stern closure

The next loft increment is tracked by [MARINE-END-CLOSURE-001 #12385](https://github.com/Luminous-Dynamics/symthaea/issues/12385) on branch `research/marine-pointed-end-closure-g2e-20261115`.

Section specification schema v1 remains accepted for legacy planar-bow/planar-stern inputs. Schema v2 adds independent bow and stern end conditions:
- `planar-end-cap`: positive-breadth terminal section closed with an explicit semantic cap fan;
- `collapsed-centreline`: exact zero half-breadth at the endpoint; the port and starboard sample columns share vertex IDs and no planar end-cap fan is created.

The G2b loft artifact is version 2. It binds both closure modes, station/point counts and separate bow/stern cap-face ranges into the manifest digest. The section-family artifact is version 2 and has a second family-manifest digest that binds source/sample digests, dimensions, coordinate frame and end-condition semantics. Older v1 section-family and v1 loft artifacts remain parseable where fields default, but fail the v2 admission verifier until regenerated; no silent semantic migration is performed.

Pointed-end waterplane closure emits one triangle where a station collapses to the centreline, rather than two triangles containing a degenerate quad. The two longitudinal end conditions are independent: pointed bow/planar stern, planar bow/pointed stern, both pointed, or both planar. Two-station both-pointed hulls, collapsed interior stations and non-zero transverse samples at a collapsed end are rejected.

The current profile still requires convex sections and identical point counts. G2e is still a closed displacement-volume geometry; it is not a full external hull, deck/freeboard, transom/stem fairness, CAD B-rep, CFD or vessel approval.

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

## G2f full external envelope input contract

G2f is a separate representation from G2a/G2b. G2a/G2b describes keel-to-waterline sections and a loading/design-waterline-capped displacement volume. G2f describes the full external envelope: underwater bottom, topsides/freeboard, deck, deck-edge seams, waterline intersections, bow/stem and stern/transom regions, explicit openings and separately versioned external components.

The G2f-a typed input validator is exposed by `full_envelope`. It checks schema-v1 typed input semantics and computes a deterministic source-spec digest; it does **not** generate a surface, mesh, B-rep or displacement shell:

    cargo run --locked -p symthaea-marine-geometry --bin marine-full-envelope-validate -- docs/research/marine/fixtures/marine-full-envelope-box-001.json --output /tmp/full-envelope-report.json

Exit codes: 0 = valid input contract; 1 = incomplete/unsupported input; 2 = rejected specification; 3 = CLI, I/O or parse error. The JSON report retains the exact disposition and issue codes. A valid input contract is not a geometry-admission or naval-engineering pass.

The canonical G2f-a schema and fixture are:
- `docs/research/marine/schemas/marine-full-envelope-spec.schema.json`
- `docs/research/marine/fixtures/marine-full-envelope-box-001.json` (symmetric transom/transom known-answer input)
- `docs/research/marine/fixtures/marine-full-envelope-asymmetric-001.json` (explicitly asymmetric port/starboard input; distinct waterline references)
- `docs/research/marine/fixtures/marine-full-envelope-pointed-bow-asymmetric-001.json` (pointed bow with shared stem IDs and asymmetric midbody)
- `docs/research/marine/MARINE-FULL-ENVELOPE-G2F-A.md`

Symmetric port geometry must be explicitly declared as derived by reflection; asymmetric geometry must supply a separate port profile. At ordinary asymmetric stations the keel is one shared centreline vertex with the same point identity and coordinates on the port/starboard profiles; the remaining profile points retain distinct identities. At a pointed end, all corresponding profile points share identities and collapse to y=0. The deck is a separate surface definition. Pointed bow/stern sections may collapse to one deck crown point; they may not be forced into a rectangular deck section.

For the synthetic prismatic fixture, the analytical displacement below z=0 is 12,000 m³ and the external envelope including its declared triangular deck crown is 20,500 m³. These are fixture values, not measurements or mesh-generator results.

The next G2f implementation stages remain distinct:
1. **G2f-b:** loft bottom and topside external surfaces with named breaklines and shared seam identities.
2. **G2f-c:** close the deck, stem and transom; reconcile all surface regions and boundary loops.
3. **G2f-d:** intersect that one full external envelope with multiple loading waterplanes to derive separate displacement shells.
4. **G2f-e:** exchange the external envelope and each derived displacement shell independently through AP242 and independent CAD-kernel re-import.

No G2f-a validator result establishes self-intersection freedom, hull fairness, waterline quality, hydrostatic equilibrium, stability, strength, seakeeping, CFD, cavitation, DFM, class approval or seaworthiness.

## Independent multi-waterline sectional reference

The G2f-a2 increment adds `sectional_reference`, a second numerical path for checking displacement and waterplane properties from the full-envelope station profiles. It is deliberately independent of G2c's tetrahedral integration of a closed triangle mesh.

The request contains one validated `source_spec` and an ordered list of `loading_conditions`. Each condition supplies:
- a unique condition ID;
- a horizontal waterplane elevation in the same vessel frame (z-down; negative elevations are above the design waterplane and imply deeper immersion);
- explicit water density in kg/m³.

For each station, the implementation constructs its transverse section from the actual starboard and port profiles (or derives port by reflection only when the spec says symmetric), clips the polygon to the submerged half-plane, calculates section area/centroid and measures the waterline breadth. It then integrates sectional area and breadth along the vessel length with piecewise-linear trapezoidal integration. First longitudinal moments use the exact integral for a piecewise-linear sectional curve. Kahan compensated summation is used for the accumulated integrals.

Run the analytical sweep fixture:

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json --output /tmp/marine-sectional-reference.json

Exit codes: 0 = evaluated or independently verified; 1 = typed source/loading/integration/integrity rejection; 2 = CLI/I/O/JSON parse failure. The emitted receipt includes source-spec and per-condition BLAKE3 digests. Independently re-evaluate a previously saved receipt with the exact original request:

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json --verify /tmp/marine-sectional-reference.json

The verifier recomputes the exact request rather than trusting the caller-supplied output, and reports the bound sweep digest on success.

### Synthetic known answers

For the 100 m × 20 m prismatic fixture, density 1,025 kg/m³, and upright horizontal planes:
- z = -2 m: 16,000 m³ displacement, 2,000 m² waterplane, VCB = 2 m;
- z = 0 m: 12,000 m³ displacement, 2,000 m² waterplane, VCB = 3 m;
- z = +2 m: 8,000 m³ displacement, 2,000 m² waterplane, VCB = 4 m.

All three have LCB = LCF = 50 m, TCB = 0, and TPC = 20.5 t/cm. The values are analytic reference results for this prismatic synthetic hull, not real-vessel predictions.

The method uses the source's station spacing and a piecewise-linear longitudinal assumption. The result is an independent numerical reference and screening curve; it is not exact for arbitrary longitudinal curvature between sparse stations. Compare the z=0 result against G2c polyhedral hydrostatics only for matching geometry, waterplane, density, frame and scope. A disagreement is a diagnostic requiring investigation, not permission to average the two results.

### Supported and unsupported scope

G2f-a2 supports upright horizontal waterplanes intersecting the station profiles between keel and sheer. It explicitly rejects planes above the sheer rather than inventing deck/flooding geometry. It does not implement trim, heel, hydrostatic equilibrium, free-surface corrections, GZ curves, intact/damage stability criteria, appendage volume, slamming, seakeeping, resistance, propulsion or CFD. It does not generate a loading-specific displacement mesh. Those remain separate geometry, hydrostatics and validation increments.

The sectional-reference CLI also has an end-to-end integration test suite covering both the prismatic box and tapered hull fixtures, digest-bound output, rejection of non-positive density, and help/scope text. Run the focused package verification as described in the marine benchmark protocol; the current draft PR's exact-head focused result is linked from issue #12434.

### Upright mass-balance waterplane solver

For a known total mass and explicit density, the sectional reference can solve the upright waterplane elevation within a caller-supplied shallow/deep bracket:

    cargo run --locked -p symthaea-marine-geometry --bin marine-upright-mass-balance -- docs/research/marine/fixtures/marine-upright-mass-balance-box-001.json --output /tmp/marine-upright-mass-balance.json

Independently recompute the receipt from the exact request:

    cargo run --locked -p symthaea-marine-geometry --bin marine-upright-mass-balance -- docs/research/marine/fixtures/marine-upright-mass-balance-box-001.json --verify /tmp/marine-upright-mass-balance.json

The solver requires:
- explicit total target mass in tonnes and water density in kg/m³;
- explicit shallow/deep elevations in the source z-down frame;
- a bracket whose displaced masses straddle the target;
- positive mass and waterplane tolerances and a bounded iteration count;
- monotonic displaced mass from shallow to deep; non-monotone or unbracketed inputs fail closed.

For the synthetic 100 m × 20 m × 6 m box, 12,300 tonnes at 1,025 kg/m³ returns z=0 m; 10,250 tonnes returns z=1 m. These are analytical fixture results.

**Scope boundary:** this is an upright sinkage / displacement-mass balance only. It does not solve longitudinal trim, heel, LCG/TCG equilibrium, KG, tank free-surface, flooding, GZ curves, intact/damage stability or statutory criteria. Commercial packages such as Maxsurf, Orca3D and NAPA already provide much more complete loading and stability workflows; this function is an evidence-bound primitive for research and pipeline integration, not a replacement.
