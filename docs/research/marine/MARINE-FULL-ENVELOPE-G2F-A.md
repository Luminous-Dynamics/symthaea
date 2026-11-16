# MARINE-FULL-ENVELOPE-G2F-A
## Full external hull profile schema and semantic fixture

**Status:** schema/fixture contract v0.1.0  
**Parent issue:** [MARINE-FULL-ENVELOPE-GENERATION-001 #12434](https://github.com/Luminous-Dynamics/symthaea/issues/12434)  
**G2e dependency:** [MARINE-END-CLOSURE-001 #12385](https://github.com/Luminous-Dynamics/symthaea/issues/12385)  
**Schema:** [marine-full-envelope-spec.schema.json](./schemas/marine-full-envelope-spec.schema.json)  
**Synthetic known-answer input:** [marine-full-envelope-box-001.json](./fixtures/marine-full-envelope-box-001.json)

> This is an input-contract and synthetic-fixture increment. It does not generate or qualify a full external hull. No geometry or physical solver result is claimed.

## 1. Scope separation

G2a/G2b describes keel-to-design-waterline sections and closes the submerged displacement volume at z=0. G2e makes the underwater endpoint semantics explicit. Those representations remain valid for their declared purpose; neither contains the full topside/freeboard and deck envelope.

G2f introduces a **separate full-envelope input type**. It must not mutate the G2a/G2b artifact's meaning or allow a waterplane cap to be relabelled as a deck.

The full external envelope is a vessel-configuration object. A displacement shell is a derived, loading-condition-dependent control volume. One full-envelope revision can therefore generate many displacement shells for different drafts and, later, trim/heel conditions.

## 2. Coordinate and profile convention

The schema fixes metres and a right-handed local frame:
- x forward, with stations ordered from bow to stern;
- y starboard, with port represented by negative y;
- z down, with the declared design waterplane at z=0;
- positive z is below the design waterplane and negative z is above it.

Each station's starboard half-profile is ordered from keel toward the sheer/deck edge. Its points are named and carry a semantic role plus the region to the next point. Under `SYMMETRIC_ABOUT_Y0`, the port profile is derived by exact reflection and shares an explicit `MIRROR:<starboard-point-id>` identity. Under `EXPLICIT_PORT_AND_STARBOARD`, each station must supply a separate port profile; no symmetry is inferred.

Deck sections are separate from the side-shell half-profiles. They are ordered from port deck edge through camber/crown points to starboard deck edge. The deck surface is therefore not inferred by extending the underwater waterplane closure.

The design waterline is explicit and named. Its per-station intersection points must resolve to the profile's waterline points (or their declared mirrored equivalents) and must lie on the declared z=0 plane within a frozen tolerance. Mirrored point identities use the reserved `MIRROR:<starboard-point-id>` form; a mirrored reference never pretends that a separate port source point was supplied.

## 3. End conditions

Bow and stern are independent:
- `POINTED_STEM`: the end station collapses to the centreplane under the declared profile rule and shares port/starboard point identities; no planar end-cap fan is created.
- `TRANSOM`: the end is closed by an explicit transom surface and face-region identity.

The first slice only defines these semantic choices. It does not yet implement arbitrary bulbous stems, flared transoms, tunnel sterns, skegs, appendage cut-outs or superstructure intersections. Such forms must be typed unsupported/INCOMPLETE until dedicated profile and topology rules exist.

## 4. Surface inventory and boundary policy

Every admitted candidate must enumerate the regions it actually contains:
- bottom;
- topsides;
- deck;
- bow-end closure;
- stern-end closure;
- optional chine and keel breaklines.

The inventory is not an inferred feature list. The validator must reconcile it with actual station profile segments, deck sections and generated face-region assignments.

A closed external shell must have no unintended open boundary. Any opening or cut-out needs an explicit boundary loop with named point references and a supported disposition. Unknown openings, duplicate seams, missing deck sections, mismatched edge identities or a transom that does not meet the side/deck surfaces cause rejection or typed INCOMPLETE—not automatic filling.

Appendages and superstructure are versioned external components. They are not silently merged into the hull shell; their identity, source digest and assembly relation are recorded separately.

## 5. Runtime semantic validation beyond JSON Schema

JSON Schema checks shape and local field domains only. A future Rust validator must additionally check:

1. Station IDs are unique; x/L is strictly increasing, starts at 0 and ends at 1; principal dimensions are finite and positive.
2. Every station has exactly one supported starboard keel, waterline and sheer/deck-edge marker in the declared order; all profile points are finite and within dimensional envelopes.
3. Profile station point IDs are unique within their identity scope; point references resolve exactly once.
4. Symmetric mode omits explicit port coordinates and uses deterministic mirrored identities; explicit asymmetric mode requires a complete port profile with compatible semantic breaklines.
5. Mirrored profiles have equal point counts and exact reflected y coordinates; port/starboard identity is preserved at shared endpoints.
6. Design waterline point references resolve to the intended section points; z=0 and waterline continuity are checked without assuming a waterplane is a deck.
7. Deck section station IDs match the full station inventory; first/last deck points coincide with their corresponding port/starboard deck-edge points; deck crown/camber remains on the declared side of z=0.
8. Bow/stern end-condition labels agree with actual terminal profile topology; pointed stations cannot contain positive transverse breadth; transom surfaces must close with the station/deck perimeter.
9. Surface region inventory, edge-to-region declarations, component seams and boundary loops are complete, non-overlapping and bound in a manifest digest.
10. All source/profile/configuration, units, frame, dimensions, symmetry and end-condition fields are digest-bound. A stale digest or semantic change without a new manifest fails closed.

The G2f-a input is not itself a B-rep, triangle mesh or full external hull artifact. A separate output schema will be required for generated surface regions, face identity, topology and geometry digests.

## 6. Synthetic box fixture

`marine-full-envelope-box-001.json` is an intentionally simple symmetric box hull:
- length 100 m, beam 20 m, draft 6 m, freeboard 4 m;
- planar/transom bow and stern;
- rectangular underwater body with vertical topsides;
- flat deck section with a small declared crown/camber;
- two longitudinal stations with exact repeated profile control points;
- one named design waterline at z=0;
- explicit bottom, topsides, deck and both end-closure regions;
- no appendages or openings.

The fixture is an input contract test, not a produced mesh. Its exact prismatic geometry yields 12,000 m³ below z=0 and 20,500 m³ for the full external envelope including the declared 0.5 m triangular deck crown. These are analytical reference values for this synthetic profile, not generated-mesh or vessel measurements.

### Explicit asymmetry and pointed-end fixture extension

The JSON fixture set now includes:
- `marine-full-envelope-box-001.json`: symmetric transom/transom reference hull;
- `marine-full-envelope-asymmetric-001.json`: explicit port/starboard profiles with different transverse breadths and independently referenced waterline points.

The pointed-stem + explicit port/starboard combination is covered by a Rust unit fixture. It requires both sides to share exact point identities and z coordinates along the collapsed stem, while permitting independent port/starboard geometry through the midbody. This distinction prevents the validator from treating asymmetric vessels as mirrored vessels or allowing a disconnected stem seam.

## 7. Next implementation increments

- **G2f-a1:** add Rust typed spec and cross-field validator; malformed-reference, asymmetry, waterline, deck-edge and end-condition negative controls.
- **G2f-b:** loft bottom/topside surfaces with shared breakline identities and explicit port/starboard policy.
- **G2f-c:** generate deck, bow and transom surfaces with a complete face-region inventory.
- **G2f-d:** intersect the external shell with declared waterplanes and emit a separate loading-specific displacement artifact.
- **G2f-e:** qualify full-envelope and derived displacement artifacts independently through AP242 export/import and validation-property preservation.

## 8. Benchmark targets

Research targets, not achieved results:
- 100% surface inventory and boundary-loop reconciliation for every artifact admitted as a full external hull;
- zero silent symmetry inference for explicitly asymmetric inputs;
- zero unintended open/non-manifold edges for candidates declared closed;
- zero duplicate vertices at declared shared seams;
- exact agreement with box/prism geometry and displacement fixtures within a frozen tolerance profile;
- stable displacement/waterplane curves across a predeclared range of horizontal waterplanes;
- identical full-envelope geometry digest across distinct loading conditions, with distinct derived displacement-shell digests;
- preserved persistent IDs, surface-region semantics and validation properties after independent AP242 round-trip.

## 9. Claim ceiling

Schema validation proves only that an input record conforms to the schema. Semantic validation proves only the declared profile contract. Neither proves a generated hull is geometrically valid, fair, watertight, structurally adequate, hydrostatically stable, manufacturable, class-approved, seaworthy or safe to operate. Those claims require separate geometry, numerical, manufacturing, physical-test and competent-authority evidence.


## 10. G2f-a typed Rust validator and CLI (16 Nov 2026)

The current implementation branch adds `crates/domains/symthaea-marine-geometry/src/full_envelope.rs` and exports it as `full_envelope`. It provides typed Rust representations for the schema-v1 full-envelope input, a deterministic source-spec digest and a cross-field validator.

The validator checks:
- finite positive dimensions, metres and the declared right-handed x-forward/y-starboard/z-down frame;
- unique station identities and strictly increasing station coordinates;
- named keel, bottom/bilge/chine, waterline and sheer profile semantics;
- symmetric-derived vs explicitly supplied port geometry;
- pointed-stem endpoint collapse and transom positive breadth;
- exact named waterline point resolution at every station;
- separate deck-section inventory and port/starboard sheer seam identity;
- required bottom, topside, deck, bow and stern surface inventory;
- opening identity, boundary-point references and supported/unsupported dispositions;
- separately versioned external component IDs and source digests.

The validator uses three dispositions:
- `ValidInputContract`: the typed source record passes the implemented input checks;
- `Incomplete`: the input explicitly requires an unsupported feature or opening;
- `Rejected`: the input violates a required semantic or integrity condition.

It deliberately does not infer geometry, fill gaps, repair profiles, transform coordinate frames, mirror explicitly asymmetric hulls or convert units. It cannot detect surface self-intersection or guarantee that the declared face topology can be lofted; those are later generator/admission stages.

CLI:

    cargo run --locked -p symthaea-marine-geometry --bin marine-full-envelope-validate -- docs/research/marine/fixtures/marine-full-envelope-box-001.json --output /tmp/full-envelope-report.json

Exit codes: 0 valid input contract, 1 incomplete, 2 rejected, 3 CLI/I/O/parse failure.

### Adversarial validation additions

The Rust tests include malformed and negative controls for missing port profiles, asymmetric input falsely declared symmetric, invalid waterline references, wrong waterline plane, incorrect deck seams, pointed-end geometry, openings whose boundary points do not resolve, missing surface regions, non-finite coordinates, unsupported regions, duplicate station/component IDs, unknown nested JSON keys and resource-bounded station counts.

The symmetric box input and the explicit asymmetric box fixture are separate cases. The asymmetric fixture supplies port and starboard profiles with different transverse breadths, distinct side point IDs and independent waterline references. The validator checks compatible semantic roles and breaklines without imposing reflected coordinates. At a pointed stem, both profiles must share exact centreplane point identities and z coordinates so that the stem seam is closed.

The symmetric box and explicit asymmetric fixture pass the typed validator on the feature branch; a pointed-stem/explicit-port case is covered by a separate unit test. A valid input contract remains strictly below generated-surface and engineering qualification.

## 11. Research benchmark programme — match the 2026 literature, exceed it through evidence

A credible comparison should not use a single neural-generation metric. Current literature already compares GAN and VAE-point-cloud-transformer generation, including geometry-distribution and diversity measures, followed by CAD reconstruction and preliminary resistance evaluation. Other 2026 work explores representation-aware generation across surface point clouds, waterline sections and buttock curves, including conditional latent diffusion. These are baselines to reproduce, not techniques to dismiss.

Relevant references:
- 2026 GAN vs VAE-PCT comparison for ship-hull generation, 8,000 generated CAD hulls and distribution/diversity plus resistance evaluation: https://www.mdpi.com/2077-1312/14/18/1707
- 2026 representation-aware ship design using point-cloud, waterline-section and buttock-curve representations: https://www.mdpi.com/2077-1312/14/7/642
- 2026 hydrostatic-constrained hull reconstruction from insufficient data: https://www.sciencedirect.com/science/article/pii/S2092678225000901
- 2026 latent-space data-driven evolutionary hull optimization under limited CFD evaluation budgets: https://www.sciencedirect.com/science/article/pii/S0029801826017695
- ITTC CFD verification and validation methodology, 7.5-03-01-01 Revision 05 (2024): https://www.ittc.info/media/11950/75-03-01-01.pdf
- ITTC benchmark database for CFD validation in resistance and propulsion, 7.5-03-02-02 Revision 03 (2024): https://ittc.info/media/11956/75-03-02-02.pdf
- ITTC uncertainty-analysis examples for CFD resistance and flow, 7.5-03-02-01 Revision 02 (2024): https://ittc.info/media/11954/75-03-02-01.pdf
- CAx-IF Cross-Domain Exchange for Downstream Uses, Version 1.0 (29 January 2026): https://www.mbx-if.org/home/wp-content/uploads/2026/02/rec_prac_cross-domain_exchange_v10.pdf
- CAx-IF 58J Summer 2026 implementer test round and Test Suite 1.2: https://www.mbx-if.org/home/cax/testrounds/
- CAx-IF recommended-practice index and revision archive: https://www.mbx-if.org/home/cax/recpractices/

### Frozen comparison arms

1. Reproduced GAN baseline using the published input/output representation, train/validation/test split rules and compute budget.
2. Reproduced VAE-point-cloud-transformer baseline on the same parent-hull-held-out split.
3. Representation-aware multi-modal model using point cloud, waterline sections and buttock curves, with conditional dimensions/form coefficients.
4. Conditional latent-diffusion model as a separate arm, not conflated with the VAE encoder.
5. Deterministic section/loft generator as a non-learned geometry baseline.
6. Evidence-carrying G2f pipeline: generated/reconstructed source → typed contract → external surface generation → multi-waterline displacement derivation → independent CAD exchange → specialist numerical validation.

All learned models must use a common, rights-cleared corpus, identical parent-hull-family splits, matched input dimensions, comparable compute budgets and fixed random seeds. Do not split randomly across near-identical variants from the same parent hull; this leaks geometry and creates inflated results. In addition, report which generator/decoder claims are directly reproduced and which are our own variants; do not compare paper-reported scores across different datasets or normalization conventions without a matched reproduction.

### Metric groups

**Generative distribution and diversity**
- Noise Score, Diversity Score, Jensen–Shannon Divergence, Minimum Matching Distance and Coverage (reproduced exactly when a source provides definitions).
- Bidirectional Chamfer distance, Hausdorff distance and 95th-percentile surface deviation.
- Normal deviation, curvature/fairness residual and breakline displacement.
- Invalid/unsupported sample rate and duplicate-design rate.

**Representation consistency**
- Point-cloud ↔ section ↔ buttock-curve reconstruction error.
- Waterline/buttock/section identity continuity and matched dimensions.
- Deck/shear/stem/transom seam residual.
- Design-waterline intersection consistency.
- Number and extent of unaccounted boundaries and surface regions.

**Engineering properties**
- Principal dimension errors and dimensionless-form coefficients: block coefficient, midship coefficient, prismatic coefficient, waterplane coefficient, longitudinal centres and wetted area.
- Displacement-volume and waterplane-area curves over multiple loading drafts.
- LCB/VCB and waterplane moments for each derived loading condition.
- Early-stage Holtrop–Mennen resistance only within its stated operating/calibration domain; use it as a screening baseline, not ground truth.
- Higher-fidelity resistance, propulsion and wake observables against ITTC-validated CFD and experimental datasets.

**Robustness and decision utility**
- False-feasible rate under hard geometry/engineering constraints.
- Calibration/coverage of bounded geometry alternatives and downstream observable intervals.
- Rank-reversal rate between low- and high-fidelity evaluators.
- Validated Pareto hypervolume/regret only after higher-fidelity/physical verification.
- Cost per generated candidate, cost per valid geometry, cost per independently validated improvement and cost to confirmed Pareto frontier.
- Reproduction rate across seeds and deterministic geometry digests.

### Competitive target ladder (research targets; not achieved results)

- **100%** explicit region inventory and boundary-loop accounting for artifacts labelled full external hull.
- **0** silent geometry repairs or inferred unit/frame transformations.
- **0** false full-hull labels for waterplane-capped G2b displacement shells.
- **100%** of accepted full-envelope samples reproduce their declared dimensions, frame and waterline intersections within the frozen input profile.
- **100%** of generated displacement-shell artifacts bind to one full-envelope revision and one explicit loading condition.
- **0** unsupported topology promoted to valid; unsupported features remain INCOMPLETE.
- Exact analytical box/prism known-answer agreement within predeclared tolerances.
- Better representation consistency and lower geometry-invalid rate than reproduced GAN/VAE-PCT baselines at matched compute budget.
- Better validated design improvement per high-fidelity evaluation than conventional evolutionary, latent-space and multi-fidelity baselines.
- Demonstrated CFD/experimental improvement on held-out parent-hull families; no result is SOTA until reproduced by an independent evaluation path.

### Why this can outperform current approaches

The potential advantage is not that Symthaea generates more shapes than a transformer or diffusion model. It is that every proposed hull can carry its geometry semantics, uncertainty, configuration identity, derived loading models and verification evidence into the same evaluation pipeline. A generator that produces fewer shapes but yields more independently validated improvements per engineering dollar can be more useful than one with better novelty or coverage scores.

The key experimental question is therefore:

**At equal compute and engineering-evaluation budgets, which system produces the most independently verified Pareto improvements while maintaining zero false-feasible geometry/physics claims?**

This must be tested. It is not an assumed advantage.
