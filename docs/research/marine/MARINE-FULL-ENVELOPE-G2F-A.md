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
- `marine-full-envelope-asymmetric-001.json`: explicit port/starboard profiles with different transverse breadths and independently referenced waterline points;
- `marine-full-envelope-pointed-bow-asymmetric-001.json`: explicit asymmetric midbody with a pointed bow whose port/starboard stem samples share exact IDs and coordinates.

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
- 1-Nearest-Neighbor Accuracy (1-NNA) as a supplementary extension; the 2026 GAN/VAE-PCT paper itself identifies 1-NNA as a future addition, so do not attribute it to that study's reported results.
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


## 12. Industry capability comparison — current product baselines

The nearest industry comparators are established naval-architecture packages, not only generative-model papers. Their published product descriptions show that the bar is already high:

| Product | Published capability baseline | What we must compare |
|---|---|---|
| **Bentley Maxsurf** | Parametric 3D vessel model, trimmed NURBS/mesh/point data, fairing and curvature, hydrostatics, intact/damaged/partially flooded stability, compartment editor, regulatory criteria, resistance methods, motions, structural modeling and CAD exchange | Hull reconstruction/fairness, offsets and coefficients, loading cases, hydrostatics, stability, resistance, geometry exchange |
| **NAPA / NAPA Designer** | 3D hull form design, parameterized variations, hydrostatics, intact and damage stability, SOLAS support, hydrodynamic analysis, powering/propulsion workflows, rule-compliance and operational calculations | Same exact hull, loading, subdivision, trim/heel and statutory calculation cases; independent outputs from a qualified NAPA build |
| **Orca3D v3** | Rhino-integrated hull design/fairing, basic hydrostatics/stability, resistance/power prediction, weight/cost tracking, advanced compartmentation/tank loads and intact/damaged stability; its October 2026 v3.2 announcement adds automated wave and self-propelled CFD setup | Model-editing workflow, real-time hydrostatics, damage/load-case semantics, model-test correlation and CFD setup/replay |
| **HydroComp NavCad** | More than 40 resistance methods, monohull/catamaran and planing/displacement cases, appendage/wave/shallow-water allowances, hull–propulsor interaction, power/efficiency, propeller sizing and model-test/sea-trial correlation | Resistance prediction across regimes, propulsion equilibrium, wake/propulsor coupling, calibration and model-test/sea-trial residuals |

Official product references:
- Bentley Maxsurf: https://www.bentley.com/en/products/maxsurf/
- NAPA naval architecture and calculations: https://www.napa.fi/software-and-services/ship-design/software/napa/
- NAPA hull-form design and hydrodynamics: https://www.napa.fi/software-and-services/ship-design/hull-form-design-and-hydrodynamics/
- Orca3D capabilities and modules: https://orca3d.com/pages/capabilities
- Orca3D v3.2 (6 October 2026): https://orca3d.com/blogs/news/orca3d-version-3-2-more-powerful-cfd-smarter-workflows-and-better-control-of-your-analysis
- HydroComp NavCad: https://www.hydrocompinc.com/solutions/navcad/

These are vendor-published capability descriptions, not independent product performance benchmarks. For a fair comparison, freeze versions, configurations, input models, criteria, solver settings and licensing/compute costs. Do not infer that every advertised method is applicable to every hull class or operating regime.

### Where this project is behind the industry today

G2f-a1 validates input semantics; it does not yet generate the external hull. G2e generates a bounded convex waterplane-capped displacement-volume mesh and is not a full external hull, fair B-rep, compartments model or statutory stability solver. The industry products above already provide much of the mature end-user functionality that our current crate does not.

The programme must not claim feature parity or a more accurate hydrostatics/stability/resistance result until the identical cases have been run against a frozen, independent product build and benchmark corpus.

### Narrow areas where we can plausibly create a differentiated advantage

1. **Evidence lineage:** every source profile, revision, transform, surface region, loading shell, solver input/output and decision is tied to exact source digests, persistent IDs, tool versions, tolerances and independent verification.
2. **External-envelope / loading-shell separation:** maintain one full external hull revision while deriving many waterline-, draft- and later trim/heel-specific displacement shells without mutating the external source geometry.
3. **Uncertainty as first-class geometry:** retain alternative reconstructions and bounded profile/survey uncertainty through geometry, hydrostatics and solver comparisons. Do not collapse epistemic reconstruction uncertainty, measurement uncertainty, numerical error and model discrepancy into one confidence score.
4. **Decision-aware multifidelity:** select the next evaluation by its potential to change a constrained engineering decision, not only by entropy reduction or surrogate-predicted improvement. Compose with the existing marine multi-fidelity/robust-Pareto/experiment-value owners rather than introducing a second optimizer.
5. **Independent qualification receipts:** no candidate is called improved merely because the same generator or surrogate that proposed it also evaluated it. Require specialist solver or physical evidence appropriate to each claimed observable.

### Competitive comparison ladder

- **C0 — shared geometry:** identical section/profile input, dimensions, frame, station positions, waterline and end conditions.
- **C1 — geometric output:** compare surface coverage, NURBS/mesh fidelity, curvature/fairness, seam residuals, boundary loops, volume, area, centroid and principal dimensions.
- **C2 — hydrostatics:** compare displacement, LCB/VCB, LCF, waterplane area and second moments over the same loading conditions and density; report residual and uncertainty for each property.
- **C3 — stability:** only after compartmentation, tanks, loads, free surfaces, downflooding points and regulatory criteria are modeled; compare intact and damage cases separately.
- **C4 — resistance/propulsion:** compare matching methods inside their supported regimes, then validated CFD and model-test/sea-trial evidence; include hull–propulsor interaction and operating-condition effects.
- **C5 — design decisions:** compare engineering time, invalid/false-feasible rate, independent-evaluation cost, and independently verified Pareto improvement per unit cost.

The desired advantage is not to beat a mature product at every calculation. It is to deliver a more reproducible, uncertainty-aware and independently auditable design/verification loop while delegating specialist calculations to qualified tools. This is a research hypothesis and requires matched evidence.


## 12. G2f-a2 — independent sectional loading-curve reference (16 Nov 2026)

The next implementation slice is `crates/domains/symthaea-marine-geometry/src/sectional_reference.rs`, exposed as `sectional_reference`, with a matching command-line binary `marine-sectional-hydrostatics-reference`.

This is intentionally a second numerical method, not a competing hydrostatics owner:
- G2c #12366/#12369 integrates oriented tetrahedra from a waterplane-capped displacement mesh.
- G2f-a2 clips full-envelope station polygons at a declared horizontal waterplane, computes sectional submerged area and section centroid, computes port/starboard waterline breadth, and integrates the curves longitudinally with a piecewise-linear trapezoidal rule.
- First longitudinal moments use the exact interval integral for the piecewise-linear sectional curve; accumulated sums use Kahan compensation.
- Results bind the exact source-spec digest, loading-condition ID, waterplane elevation, explicit density and an independent result digest. The complete sweep has its own digest and a fresh recomputation verifier.

### Request and CLI

`docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json` contains the full-envelope input and three ordered upright loading conditions.

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json --output /tmp/marine-sectional-reference.json

The v1 loading contract uses a horizontal plane in the source frame, z-down, and an explicit positive density in kg/m³. A waterplane elevation of -2 m is 2 m above the design waterplane in the vessel frame and represents deeper immersion; +2 m is a shallower loading. Conditions must be uniquely identified and ordered by ascending z. Density is never inferred.

### Synthetic known-answer results

The 100 m × 20 m prismatic box hull has analytic values:
| Waterplane z | Displacement | Waterplane area | VCB |
| ---: | ---: | ---: | ---: |
| -2 m | 16,000 m³ | 2,000 m² | 2 m |
| 0 m | 12,000 m³ | 2,000 m² | 3 m |
| +2 m | 8,000 m³ | 2,000 m² | 4 m |

For all three conditions, LCB = LCF = 50 m, TCB = 0, and at 1,025 kg/m³ TPC = 20.5 t/cm. These are analytical synthetic results. They are not measured vessel data.

An explicit asymmetric-port negative/positive fixture uses the actual port profile rather than silently mirroring it. The analytic design-waterplane result is 11,400 m³ and TCB = +0.5 m for a 10% narrower port half-section than starboard. Pointed ends can contribute zero area at their terminal stations while remaining valid longitudinal integration endpoints.

### Validation and claim boundary

The request must first pass the G2f full-envelope input contract. The method only supports upright horizontal waterplanes that intersect every station profile between keel and sheer. It rejects a waterplane above the sheer instead of inventing deck immersion/flooding. It does not produce a displacement mesh and does not support trim, heel, free-surface corrections, equilibrium solving, GM/GZ stability, damaged compartments, appendage volume or class approval.

The longitudinal integral is piecewise linear between the supplied stations. It is exact for the prismatic known-answer fixture and is an independent reference for station-fair hulls only to the degree that the station spacing resolves their sectional-area and waterline-breadth curves. Sparse station sets must be refined and convergence reported. A mismatch with G2c is a diagnostic; do not average the methods or select the preferred answer without identifying the source of the discrepancy.

### Composition with existing owners

This reference composes with:
- [G2c hydrostatics #12366 / implementation PR #12369](https://github.com/Luminous-Dynamics/symthaea/issues/12366);
- [G2d finite-difference sensitivity #12383 / PR #12384](https://github.com/Luminous-Dynamics/symthaea/issues/12383);
- [MARINE-ENVELOPE-001 #8130](https://github.com/Luminous-Dynamics/symthaea/issues/8130) for frozen operating-condition evidence coverage;
- [MARINE-MF-001 #8138](https://github.com/Luminous-Dynamics/symthaea/issues/8138) and [validated multifidelity design #11513](https://github.com/Luminous-Dynamics/symthaea/issues/11513) for cost and fidelity selection;
- [MARINE-HYDRODYNAMIC-VALIDATION-001 #11442](https://github.com/Luminous-Dynamics/symthaea/issues/11442) for specialist solver V&V.

It is a geometry-derived reference and evidence adapter, not a new hydrostatic stability package, experiment selector or CFD implementation.


## 13. G2f-a2b — tapered-hull analytical and station-refinement fixture (16 Nov 2026)

A second known-answer input is `marine-full-envelope-tapered-loading-sweep-001.json`. Unlike the box fixture, it is non-prismatic: length 100 m, full beam increases linearly from 10 m to 20 m from bow to stern, draft 6 m, freeboard 4 m, transom ends, and symmetric sections. It uses the same explicit density of 1,025 kg/m³ and upright horizontal waterplanes z=-2, 0 and +2 m.

Analytical results:
| Waterplane z | Displacement | Waterplane area | TPC | LCB / LCF | VCB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -2 m | 12,000 m³ | 1,500 m² | 15.375 t/cm | 55.5556 m | 2 m |
| 0 m | 9,000 m³ | 1,500 m² | 15.375 t/cm | 55.5556 m | 3 m |
| +2 m | 6,000 m³ | 1,500 m² | 15.375 t/cm | 55.5556 m | 4 m |

TCB is zero by symmetry. Design-waterplane displaced mass is 9,225 tonnes.

A second unit test refines this same linear taper from 2 to 3 to 5 longitudinal stations. Because section area and waterline breadth are linear in x for this controlled fixture, the piecewise-linear integration should preserve displacement, waterplane area, LCB and VCB within a frozen 1e-9 absolute test tolerance. The source-spec digest must differ after refinement even though the physical numeric answers remain the same. This is a test of deterministic identity and integration invariance for a known linear case; it is not evidence that arbitrary curved hulls have converged. For curved/nonlinear stations, perform systematic refinement studies and report the change in each observable instead of assuming linear interpolation is exact.


## 14. Published generative-hull baseline: quantitative CAD recovery and shape-distribution targets

A useful current baseline is the September 2026 GAN versus VAE-PCT hull-generation comparison (JMSE 14(18), 1707). It reports 7,795 barge-type and 8,498 containership training CAD hulls, then evaluates 8,000 generated CAD outputs per generator and hull category. The pipeline pulls discrete generator outputs back to continuous B-spline surfaces by fairing section curves and lofting the results. This is already a meaningful CAD-to-CAD generative baseline; simply producing point clouds or lofted surfaces is not a sufficient novelty claim.

The paper evaluates five complementary metrics: Noise Score (NS), Diversity Score (DS), Jensen–Shannon Divergence (JSD), Minimum Matching Distance (MMD) and Coverage (COV). It reports the following pull-back failures:

| Hull family | GAN failures | VAE-PCT failures |
| --- | ---: | ---: |
| Barge-type | 441 / 8,000 (5.5%) | 126 / 8,000 (1.6%) |
| Containership | 361 / 8,000 (4.5%) | 194 / 8,000 (2.4%) |

The trade-offs matter: VAE-PCT generally produces less noisy geometries and performs better on the more complex containership set, while GAN generally achieves greater coverage of the training design space. These results should be reproduced on the published definitions and matched splits, not compressed into a universal winner label.

The paper also finds systematic underestimation of midship-area coefficient C_M, with upward bias in derived prismatic coefficient C_P and vertical centre of buoyancy ratio VCB/T. This is an important warning: a generator can reproduce overall point-cloud distributions yet distort naval-architectural quantities that affect design decisions.

### Our targets against that baseline (research targets, not achievements)

- **CAD pull-back success ≥99%** on each predeclared held-out hull family, improving on the reported 94.5–98.4% success range in the four baseline cases.
- **No silent repair:** every fairing, healing, resampling or coordinate change is separately reported and digest-bound. Report source-to-CAD deviations before and after any explicitly permitted operation.
- **No geometry-only victory:** add errors in displacement, C_M, C_P, C_WP, LCB, LCF, VCB/T, waterplane area and wetted surface, not just point-cloud similarity.
- Report NS, DS, JSD, MMD and COV using the paper's definitions and normalization; add 1-NNA as a supplementary metric, noting that the 2026 paper lists it as a future extension rather than a reported baseline result.
- Use parent-hull-family holdouts and hull-class holdouts. Do not randomly split nearly identical geometry variants across train and test.
- Add robustness cases for unusual stems, transoms, asymmetry, deck camber, sparse stations, reconstruction holes, high curvature and geometry outside the training distribution.
- Report 95% confidence intervals, compute budget, generation count, pull-back failure denominator, invalid/unsupported rate and cost per independently validated candidate.

The target is not merely to reduce failure below 1%; it is to reduce failure while preserving or improving the baseline's diversity and design-space coverage, and while preventing geometric bias from becoming a false hydrostatic or hydrodynamic claim.

## 15. The multimodal-generation novelty boundary

Representation-aware generative ship design is already present in the 2026 literature: a conditional multimodal autoencoder learns a shared latent representation of surface point clouds, waterline sections and buttock curves, then a conditional latent diffusion model generates representations from principal dimensions and form coefficients.

Consequently, we should not claim that combining point clouds, sections and buttock curves is itself the paradigm shift. The differentiator must be what happens *after* the proposal:
- persistent identities and cross-representation consistency;
- full external envelope and loading-state separation;
- explicit source and reconstruction uncertainty;
- CAD/STEP semantic preservation with downstream traceability;
- independent hydrostatic and hydrodynamic verification;
- experiment selection by decision impact;
- robust mission-level Pareto decisions with hard safety gates.

This is a more demanding comparison than matching a training distribution. It asks whether a candidate remains an identifiable, physically coherent and decision-useful engineering object through the whole design-to-evidence lifecycle.

## 16. G2f-a3 — shared keel identity in asymmetric section profiles

For explicitly asymmetric port/starboard sections, the keel is a common centreline vertex. Both profiles must share its point identity and coincident y=0/z coordinates. Other non-pointed profile points remain separately identified. At a pointed bow or stern, all corresponding terminal profile points share identity and collapse to y=0.

This distinction is now enforced by the typed validator and tested against split keel identity and mismatched keel elevation. It matters for later G2f-b shell construction: the independent reference must not close a polygon by connecting two nominally different keel vertices whose z coordinates differ.

Latest focused evidence is recorded in issue #12434 and PR #12438; it remains focused crate verification, not full workspace CI or physical qualification.


## 17. Nonlinear station-resolution convergence benchmark (G2f-a4)

The current linear-taper fixture is analytically exact under piecewise-linear longitudinal integration. It proves implementation consistency but does not measure discretization convergence. A second, intentionally nonlinear known-answer hull is therefore included in `sectional_reference.rs` tests.

### Frozen analytic shape

A 100 m prismatic-depth hull has half-breadth:

`b(u) = 8 + 4u(1-u)` metres, where `u = x/L` and `0 ≤ u ≤ 1`.

Draft is 6 m; the section is rectangular below the z=0 waterplane. Therefore:

- sectional area: `A(u) = 12 b(u)` m²;
- exact displacement: `L ∫₀¹ A(u) du = 10,400 m³`;
- exact waterplane area: `L ∫₀¹ 2b(u) du = 5,200/3 m²`;
- LCB = 50 m by longitudinal symmetry;
- TCB = 0 and VCB = 3 m at z=0.

The geometry and condition are kept fixed while only the station sampling density changes. Three samplings are compared: 3, 5 and 9 equally spaced stations.

### Expected trapezoidal integration convergence

| Stations | Displacement estimate (m³) | Absolute error (m³) | Waterplane estimate (m²) | Absolute error (m²) |
|---:|---:|---:|---:|---:|
| 3 | 10,200.0 | 200.0 | 1,700.0000 | 33.3333 |
| 5 | 10,350.0 | 50.0 | 1,725.0000 | 8.3333 |
| 9 | 10,387.5 | 12.5 | 1,731.2500 | 2.0833 |
| Analytic | 10,400.0 | 0 | 1,733.3333 | 0 |

Doubling interval count reduces the error by exactly four for this quadratic shape, demonstrating the expected second-order convergence of the composite trapezoid rule. Tests assert the actual errors and both 4× ratios; they do not infer convergence from a plot or paper claim.

### What this establishes and what it does not

It establishes numerical quadrature behavior for a known analytic longitudinal sectional-area / waterline-breadth field. It does not establish that a sparse station set reconstructs an unknown real hull, that station interpolation is physically accurate, or that this method converges for discontinuous/transom/wave-making phenomena without additional treatment.

For real vessels, publish the original station data, sampling policy and a nested-resolution comparison from the same frozen CAD/survey source. Compare displacement, waterplane area, LCB/LCF, VCB and sectional-area curve changes; record the denominator when a reference value approaches zero. Only call a result converged when the refinement residual is below a predeclared tolerance and remains stable over at least two consecutive refinements.

The sectional reference CLI now has `--verify <receipt.json>`. That mode re-evaluates the exact input request, compares the complete receipt and returns a typed integrity rejection on mismatch. It verifies reproducibility of this particular algorithm/request; it is not an independent implementation. The separately authored sectional method remains independent from the G2c polyhedral mesh integration path.


## 18. Upright mass-balance solver and industrial comparison (G2f-a5)

### Implemented research increment

`sectional_reference.rs` now includes `solve_upright_mass_balance` and `verify_upright_mass_balance_receipt`, plus the `marine-upright-mass-balance` CLI.

The request explicitly binds:
- full-envelope source spec and its BLAKE3 digest;
- target vessel mass in tonnes;
- explicit water density in kg/m³;
- shallow and deep waterplane elevations in the source z-down frame;
- mass and plane tolerances;
- bounded bisection iteration count.

Before solving, it checks source-contract validity, bracket ordering, endpoint waterplane support, positive density/tolerances, and monotonic displacement (shallower plane must not displace less mass than deeper plane). Unbracketed targets and exhausted iteration budgets are rejected. The receipt binds the initial/final bracket, selected waterplane, derived sectional properties, mass residual, iteration count and stop reason. The verifier recomputes the exact request.

### Analytical known answers

For the 100 m × 20 m prismatic box hull at 1,025 kg/m³:
- target mass 12,300 t → displacement 12,000 m³ → waterplane z=0 m;
- target mass 10,250 t → displacement 10,000 m³ → waterplane z=1 m.

The first case solves at the first bisection midpoint; the second at the second midpoint. An out-of-bracket mass, zero density, invalid tolerance or iteration budget is rejected. The JSON fixture is tested directly.

### Industrial capability boundary

These are not empty-market capabilities. Commercial naval-architecture products already cover much more than the current G2f-a reference:
- Bentley Maxsurf documents trimmed NURBS hull modelling, fairing, curvature analysis, point-data fitting, transformation to target hydrostatics, and downstream stability analysis including intact/damaged conditions, load cases and strength-related calculations. <https://www.bentley.com/en/products/maxsurf/>
- Orca3D Advanced Hydrostatics & Stability documents compartments, fixed/fluid loads, loading cases, free-surface effects, equilibrium flotation, stability curves and criteria evaluation. It describes three-dimensional volume integration from the surface model. <https://orca3d.com/pages/advanced-hydrostatics-stability>
- NAPA documents hydrostatics, capacity and loading conditions, intact/damage stability and statutory/rule-compliance workflows. <https://www.napa.fi/software-and-services/ship-design/hydrostatics-and-stability/>

Therefore, we must **not** claim that the G2f sectional reference or upright mass-balance solver matches those packages in scope or replaces them. Its present role is a transparent, independently testable reference and evidence-carrying adapter.

### Where the competitive opportunity actually is

The 2026 generative-design literature already has:
1. GAN and VAE-PCT hull generation, a CAD pull-back step to B-spline surfaces, 8,000 processed generated CAD hulls, NS/DS/JSD/MMD/COV evaluation, nine nondimensional coefficients and an early-stage resistance estimate. <https://www.mdpi.com/2077-1312/14/18/1707>
2. Representation-aware conditional multimodal autoencoding across surface point clouds, waterline sections and buttock curves, coupled to conditional latent diffusion. The reported experiments use a synthetic hull dataset and claim coherent controllable hull generation. <https://www.mdpi.com/2077-1312/14/7/642>

Simply claiming a multimodal latent representation or diffusion model would therefore not distinguish our work. Nor would claiming that we invented CAD-to-CAD hull generation.

The potential differentiation is a complete **design-to-decision evidence chain**:
- source drawing/survey/CAD provenance and rights status;
- reconstruction alternatives and uncertainty, rather than one falsely exact fitted surface;
- persistent identities across sections, point clouds, surfaces and CAD exchange;
- one external-envelope revision producing many loading-specific displacement and condition artifacts;
- independent sectional and polyhedral hydrostatics, followed by qualified commercial/open solver comparison;
- cross-kernel AP242 geometry *and semantic* recovery;
- cost-aware high-fidelity/physical experiment selection;
- robust mission-level selection in which non-compensable safety/service constraints cannot be offset by a weighted score.

This is a hypothesis to test, not a proven superiority claim.

### Fair benchmark plan against SOTA and industry

| Comparator | What we must reproduce or measure | Our proposed competitive outcome |
|---|---|---|
| 2026 GAN / VAE-PCT | NS, DS, JSD, MMD, COV; same data splits and normalization; CAD pull-back success; geometry coefficient distribution | Higher valid CAD recovery and engineering-feasible sample fraction without losing diversity/coverage |
| 2026 multimodal autoencoder + latent diffusion | Point clouds, waterlines and buttocks; shared-latent cross-modal reconstruction; controllability by principal dimensions/form coefficients | Better exact identity, seam and hydrostatic consistency after CAD round-trip and loading derivation |
| Maxsurf | Surface fit/fairing, target-hydrostatics matching, intact/damage load-case outputs where licensed test cases are available | Independent reproducibility, source-to-result traceability and cross-tool evidence preservation—not solver breadth |
| Orca3D | Surface-volume hydrostatics, equilibrium, tank/free-surface and load-case comparison where the same supported cases can be constructed | Exact geometry/configuration binding and independent cross-implementation receipts—not a claim to its stability capability |
| NAPA | Hydrostatics, loading/capacity and rule workflows where reference cases/data are available | Evidence provenance, rights-aware historical reconstruction and multi-tool reproducibility—not a claim to statutory approval |
| ITTC CFD/EFD | Series 60 and published benchmark database cases, mesh/grid verification and uncertainty per observable | No CFD-performance claim until each observable is independently validated within declared combined uncertainty |

Primary V&V baselines:
- ITTC 7.5-03-01-01 Revision 05 (2024), *Uncertainty Analysis in CFD Verification and Validation*: https://www.ittc.info/media/11950/75-03-01-01.pdf
- ITTC 7.5-03-02-01 Revision 02 (2024), *Uncertainty Analysis in CFD, Examples for Resistance and Flow*: https://ittc.info/media/11954/75-03-02-01.pdf
- ITTC 7.5-03-02-02 Revision 03 (2024), *Benchmark Database for CFD Validation for Resistance and Propulsion*: https://www.ittc.info/media/11956/75-03-02-02.pdf
- CAx-IF current recommended-practice index and revision archive for persistent IDs, AP242 validation properties, assemblies and STEP/QIF trace cases: https://www.mbx-if.org/home/cax/recpractices/ and https://www.mbx-if.org/home/cax/recpractices/archive/

### Benchmark targets to aim for

These are programme targets, not measured achievements:
- **100%** of accepted geometry/physics receipts bind exact source/configuration, algorithm, units/frame, solver version or method, assumptions and input/output digests.
- **0** false-feasible geometry or unsupported-feature promotions; unsupported profile regions remain INCOMPLETE.
- **0** silent unit, coordinate-frame, symmetry, topology or surface-healing operations.
- **100%** exact analytical agreement on frozen box/prism fixtures within predeclared tolerances.
- Demonstrated second-order station convergence on smooth quadratic sectional properties; for actual hulls, two consecutive refinements below a frozen observable-specific tolerance.
- **100%** of derived loading artifacts identify one source-envelope revision, waterplane and density.
- Cross-kernel STEP AP242 round-trip: independently re-import, compare exact or tolerance-bounded geometry, persistent IDs, semantic face regions, product structure and validation properties.
- On matched held-out hull families, lower invalid/unsupported sample rate and higher valid-CAD recovery than reproduced GAN/VAE-PCT/latent-diffusion baselines at equal compute.
- On matched high-fidelity budgets, higher independently validated Pareto improvement per solver-hour/test-dollar than conventional parameterized, evolutionary, and surrogate-assisted baselines.
- For physical-performance claims, no candidate wins on low-fidelity or generated labels alone; report false-feasible rate, calibration/coverage, rank reversal, validated regret and cost to confirmed frontier.

### What must not be claimed

A valid full-envelope input contract is not a valid hull surface. A closed displacement volume is not a full external hull. A mass-balanced upright waterplane is not a trim/heel equilibrium. Hydrostatics is not stability approval. A correct CAD export header is not a valid B-rep. A low CFD residual is not experimental validation. A Pareto point is not a safe, class-approved or seaworthy vessel.

The first competitive milestone should be a reproducible, rights-cleared public benchmark in which an external evaluator can regenerate the geometry, independently recover the CAD, recompute multi-waterline properties, and inspect every provenance/uncertainty boundary without trusting Symthaea's own generator or solver.


## 12. G2f-a6 — make solver settings part of upright mass-balance evidence (16 Nov 2026)

The upright mass-balance receipt is now schema v2 / algorithm `upright-mass-balance-bisection-v2`. It binds:
- mass residual tolerance (tonnes);
- waterplane bracket tolerance (metres);
- requested maximum iteration budget;
- initial and final waterplane brackets;
- exact full-envelope source digest, load case, target mass and water density;
- equilibrium properties, residual, actual iteration count and stop reason.

These numerical stopping parameters are included in the BLAKE3 receipt identity. A regression test solves the same exact midpoint load case with different requested iteration budgets and verifies that, even when both runs produce the same equilibrium plane, their receipts differ. The receipt schema version is advanced explicitly rather than silently changing a v1 evidence contract.

The CLI's large receipt variant is boxed to keep the command-line report enum compact. No Clippy lint is suppressed.

This change closes a subtle reproducibility gap: the final result alone does not fully describe how it was accepted. Numerical tolerances and evaluation budgets are part of the engineering evidence.

## 13. Static attitude equilibrium — next new workstream

Created [MARINE-STATIC-ATTITUDE-EQUILIBRIUM-001 #12498](https://github.com/Luminous-Dynamics/symthaea/issues/12498).

The current upright bisection solves only one unknown: waterplane offset for a known mass. A vessel with a longitudinal or transverse CG offset also needs trim and heel. The next solver should find the static pose using:
- force residual: displaced mass minus declared vessel mass;
- two independent moment residuals from `(r_B-r_G) × g_b`;
- exactly three unknowns: waterplane offset, pitch and roll; calm-water yaw is a gauge/fixed variable;
- an oblique waterplane in the body frame, `g_b · r = c`, with section clipping performed against the true plane rather than approximated by LCB/TCB at an assumed upright gravity direction.

This is not the same as the existing WAPS 3-DOF manoeuvring/leeway/rudder coupling. It is also not a GZ/stability solver. An equilibrium can be unstable; intact/damage stability, free-surface/tank, downflooding and statutory criteria must remain separate.

The issue includes a bounded 3-variable solver contract, body-frame equations, analytical box/wedge/asymmetric fixtures, cross-checks against the existing polyhedral G2c path and matched Maxsurf/NAPA cases. It explicitly rejects unsupported deck immersion/flooding and non-convergence.

## 14. Reassessed 2026 SOTA and industry comparison (verified 16 Nov 2026)

### Generative hull design

Current research already covers several techniques that might otherwise look like novel Luminous directions:

- **Representation-aware generative ship design (Htein et al., JMSE 2026, 14(7), 642):** shared latent encoding from surface point clouds, waterline sections and buttock curves, a conditional latent diffusion generator and surface reconstruction. The authors report improved geometric coherence and controllability on a synthetic dataset. This means multi-representation latent generation by itself is not a defensible novelty claim. Source: https://www.mdpi.com/2077-1312/14/7/642
- **GAN vs VAE-PCT ship hull generation (JMSE 2026, 14(18), 1707):** a CAD-to-CAD workflow processes 8,000 generated CAD hulls, pulls discrete outputs back to B-spline surfaces via lofting/fairing and evaluates geometric validity, non-dimensional form coefficients and early-stage calm-water resistance. This is a strong reproduction baseline, not an industry-wide performance guarantee. Source: https://www.mdpi.com/2077-1312/14/18/1707
- **Offline data-driven evolutionary hull optimization (Ocean Engineering, 15 June 2026, 125935):** searches a learned latent geometry space under strict CFD evaluation budgets and directly addresses parent-hull dependence and surrogate-then-optimize bias. We must compare against this architecture, not just conventional parametric variation. Source: https://www.sciencedirect.com/science/article/pii/S0029801826017695
- **Hydrostatic-constrained reconstruction from insufficient data (IJNAOE 2026, 18, 100732):** reconstructs hull forms using hydrostatic constraints, area-constrained waterlines and curve-fairness terms; the study evaluates container, bulk-carrier and twin-skeg examples. Hydrostatic constraints in reconstruction are established research and must be included as a baseline. Source: https://www.sciencedirect.com/science/article/pii/S2092678225000901
- **Experimental toroidal propulsor co-design (Engineering Applications of AI, 15 Nov 2026, 116126):** combines open-water experiment-validated CFD, 14 design variables and adaptive RBF/evolutionary search. It reports a 7.26% power reduction at matched thrust and a 1.74 dB noise reduction relative to its best sampled design. This is study-specific. It should be reproduced as an architecture-specific comparator, not treated as a generic expected gain. Source: https://www.sciencedirect.com/science/article/pii/S0952197626024103

### Commercial engineering capability

The programme must be honest about current product capability:

| System | Documented capability | Our comparative opportunity |
|---|---|---|
| Bentley Maxsurf Stability | Direct trimmed-NURBS integration, loading conditions, equilibrium, GZ, intact/damage stability, tank/compartment modelling, criteria and longitudinal strength | Evidence lineage, reproducible data exchange and open independent validation; do not claim feature parity |
| NAPA | Comprehensive naval-architecture calculations and stability; NAPA 2026.1 prepared Designer/Engineer for AI-assisted workflows | Bind approved hull/compartment configurations to digital-twin/loading workflows, preserving revision and evidence provenance |
| Siemens Simcenter STAR-CCM+ | Transient multiphase free-surface VOF, DFBI and six-degree-of-freedom ship motion; full hull/propulsion/rudder/appendage interaction workflows | Integrate specialist solvers; compete on validated decision efficiency, not on an unvalidated home-grown CFD solver |
| CAx-IF / STEP AP242 ecosystem | Managed-model exchange, product structure, persistent IDs and validation-property recommended practices | Demonstrate semantic round-trip fidelity across actual independent CAD kernels, not just a valid STEP header |

Sources:
- Maxsurf Stability: https://maxsurf.net/stability and https://maxsurf.net/stability-capability
- NAPA: https://www.napa.fi/software-and-services/ship-design/software/napa/ ; 2026.1 https://www.napa.fi/napa-release-2026-1/ ; digital-twin/loading bridge: https://www.napa.fi/a-digital-twins-best-foundation-is-meaningful-use-of-data/
- Siemens hull-performance workflow: https://static.sw.cdn.siemens.com/siemens-disw-assets/public/4b92hDfRCo5ND51UfBtk7p/en-US/Siemens-SW-Simcenter-STAR-CCM%2B-Hull-Performance-Workflow-Fact-Sheet_tcm27-80626.pdf
- ISO 10303-242:2025 Edition 4 (published 25 August 2025): https://www.iso.org/standard/84300.html
- ISO/CD 10303-242 Edition 5 entered committee development on 23 February 2026: https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/09/32/93277.html
- CAx-IF current recommendations include Persistent IDs v1.8 (13 March 2026), Cross-Domain Exchange v1.0 (29 January 2026) and Composite Material Validation Properties v1.0 (23 June 2026): https://www.mbx-if.org/home/cax/recpractices/

### CFD verification and validation baseline

Use the applicable current ITTC procedures and freeze the revision in every test manifest:
- 7.5-03-01-01 Revision 05 (2024), uncertainty analysis in CFD verification and validation: https://www.ittc.info/media/11950/75-03-01-01.pdf
- 7.5-03-02-01 Revision 02 (2024), uncertainty analysis examples for resistance and flow: https://www.ittc.info/media/11954/75-03-02-01.pdf
- 7.5-03-02-02 Revision 03 (2024), benchmark database for CFD validation in resistance and propulsion: https://www.ittc.info/media/11956/75-03-02-02.pdf
- 7.5-03-02-04 Revision 02 (2024), practical guidelines for ship resistance CFD, as listed in the ITTC catalogue: https://www.ittc.info/search/?q=7.5-03-02-

Do not label a global solver validated because one observable or one operating point agrees. Report grid/time-step/iterative uncertainty, validation uncertainty, model discrepancy and experimental uncertainty separately, with an explicit applicability domain per observable.

### How to beat the state of the art fairly

The strongest defensible differentiation is not another generator architecture. It is the **cost to independently validated engineering improvement**.

Run matched, rights-cleared, parent-hull-family-held-out experiments:

1. Deterministic parametric/section loft baseline.
2. Reproduced GAN baseline from the published CAD-to-CAD comparison.
3. Reproduced VAE-PCT baseline.
4. Representation-aware shared-latent point-cloud/waterline/buttock generator.
5. Conditional latent-diffusion model as a separately measured arm.
6. Offline latent-space evolutionary optimization under a fixed CFD budget.
7. Luminous evidence-carrying design pipeline: geometry admission, multi-waterline curves, loading-equilibrium qualification, real AP242 round-trip, specialist CFD/EFD evidence and robust Pareto selection.

Freeze the same train/test parent-hull families, geometry normalization, dimensional constraints, operating cases, high-fidelity evaluation budget and random seeds. Near-identical sister hulls must not leak across train/test. Do not compare published headline metrics obtained on different datasets as if they were matched.

### Benchmark scorecard

| Layer | Required measurements |
|---|---|
| Generation/reconstruction | Validity rate, diversity, coverage, MMD/JSD where definitions are reproducible, Chamfer and Hausdorff/HD95, normal deviation, curvature/fairness, form-coefficient error |
| Representation consistency | Point-cloud/section/buttock/waterline round-trip error, deck/stem/transom seam mismatch, region inventory, unaccounted boundaries |
| Hydrostatics/loading | Displacement and waterplane curves, LCB/TCB/VCB, waterplane moments, TPC, upright mass residual, then trim/heel force and moment residuals |
| CAD interoperability | Independent kernel import/export, B-rep validity, geometric deviation, persistent-ID continuity, PMI/GD&T and validation-property preservation |
| Hydrodynamics | ITTC resistance/propulsion/wake observables, grid/time-step uncertainty, independent CFD/EFD discrepancy, applicability regime |
| Propulsion | Matched thrust, delivered power, rpm, advance coefficient, propulsive efficiency, cavitation and underwater radiated noise |
| Mission | Energy/GHG/lifecycle cost per delivered cargo-work, operability, availability and degradation under frozen route/loading/weather distributions |
| Evidence economics | False-feasible rate, rank reversal, validated selection regret, cost per independently verified Pareto improvement and cost to confirmed frontier |

### Target ladder — targets, not achievements

- **100%** explicit surface-region and boundary-loop accounting for full-envelope artifacts.
- **0** silent repairs, coordinate-frame/unit inference or displacement-shell/full-external-hull semantic confusion.
- **100%** loading receipts bind source hull revision, load case, density, frame, solver method and numerical stopping settings.
- **0** candidates accepted with unsupported topology, unaccounted openings or false-feasible hard constraints.
- **95%** empirical coverage of declared 95% prediction intervals on genuinely held-out hull families and operating regimes, reported separately by observable and regime; recalibrate or abstain when coverage fails.
- **10%+** mission-energy improvement as an initial target and **15%** stretch target only when independently validated on matched service and operating-envelope coverage.
- **10%+** validated Pareto hypervolume improvement and **20%** reduction in cost to reach the independently validated frontier versus matched baselines.
- **0** mandatory safety or statutory violations accepted.
- A generator must improve the number/value of independently validated designs per fixed high-fidelity/physical-evaluation budget, not just improve a generative score.

### Paradigm shifts

1. **External object vs operating-state family:** one immutable full external hull revision should support many loading waterplanes, attitudes and service cases, each with a linked derived displacement/control-volume artifact.
2. **Geometry as evidence-bearing configuration:** persistent surface/feature identity, uncertainty, tolerances and CAD validation properties travel with the geometry.
3. **Inverse design and reconstruction as one continuum:** new designs and archival/as-built reconstructions use the same declared profile semantics, with uncertainty alternatives preserved instead of forcing one falsely precise hull.
4. **Multifidelity acquisition by decision consequence:** choose the next physical or high-fidelity experiment for its expected probability of changing a feasible engineering decision, not merely maximum uncertainty reduction.
5. **Lifecycle engineering rather than one-shot optimization:** measure improvement through loading, refit, fouling/maintenance, degradation, route/weather variability and downstream service.

The falsifiable competitive hypothesis is:

**At the same compute and specialist-evaluation budget, the evidence-carrying pipeline produces more independently validated Pareto improvements, with lower false-feasible rate and lower cost to confirmed frontier, than reproduced generative and latent-space optimization baselines.**

If the matched experiment does not support this, we must report that result rather than assert superiority.


## 15. Generative-hull metric failure modes to reproduce explicitly

A closer read of the 14 September 2026 GAN/VAE-PCT paper exposes a useful falsification target: its CAD-to-CAD distribution comparison found a systematic underestimation of the midship-section coefficient `C_M`. That bias propagates to an upward shift in prismatic coefficient `C_P` and, to a smaller extent, volumetric prismatic coefficient `C_{VP}`; a mild upward tendency in VCB/T is also observed. The authors identify this as a future area for their representation/encoding scheme.

The benchmark must therefore not declare a generator strong just because its Chamfer distance and sample-distribution metrics look good. A model can reproduce plausible geometry while biasing quantities directly connected to displacement distribution and hydrostatics.

Add these scorecard requirements:
- report the full distribution, bias, variance and tail error for `C_B`, `C_M`, `C_P`, `C_{WP}`, `C_{VP}`, LCB/L, VCB/T, LCF/L and wetted surface area, with exact formula/normalization and reference data;
- add a coefficient-specific calibration plot and standardized residual per hull family;
- stratify errors by bow/stern type, midship fullness, bulb/skeg/twin-skeg class, and parent-hull family;
- include an explicit `C_M` underestimation negative control and track its propagation to `C_P`, `C_{VP}` and VCB/T;
- preserve the paper's five discrete generation metrics (Noise Score, Diversity Score, JSD, MMD, Coverage) and include 1-NNA as an additional metric where its definition can be reproduced;
- record the VAE-PCT decoder patch-size effect separately (the paper identifies patch-wise correlated errors and proposes revisiting patch size 12); and
- compare both discrete point-cloud output and post-fairing B-spline CAD output. The pull-back stage is part of the CAD-to-CAD system and cannot be hidden as a post-processing detail.

The source study's training hull databases comprise 7,795 barge-type hulls and 8,498 containership hulls. It evaluates synthetic generated hulls and pulls discrete model output back to smooth `C^2` B-spline surfaces using curve fairing and lofting. These are two bounded single-type datasets; do not treat them as a representative random sample of the global fleet.

### Raster2Hull numeric reproduction boundary

The 31 July 2026 Raster2Hull paper reports a synthetic 5,000-pair ShipPlan dataset and a template-deformation point-cloud network. Its reported GA-only reconstruction metrics include:
- global Chamfer distance: 0.499 ± 0.182 × 10³;
- global F-score @1%: 46.45% ± 8.05%;
- bow sectional F-score @1%: 58.28% ± 14.94%; and
- stern sectional F-score @1%: 60.94% ± 12.36%.

For lines-plan reconstruction the paper reports an overall F-score @1% of 50.94% ± 10.54%. Preserve the paper's exact normalization and distance units; do not compare these figures to differently normalized point clouds.

These are synthetic held-out split results. The authors explicitly state the model is not evaluated on heavily degraded real archive scans and that the generated point clouds are intended for macro-level spatial estimation, not perfectly watertight CAD. Our benchmark must therefore have three separate gates:
1. reproduce synthetic benchmark results;
2. evaluate clean and degraded real archival drawings with explicit uncertainty/abstention; and
3. compare against paired surveyed/as-built 3D ground truth before claiming real-vessel reconstruction accuracy.

Primary sources:
- GAN/VAE-PCT study: https://www.mdpi.com/2077-1312/14/18/1707
- Raster2Hull IEEE Access paper: https://xplorestaging.ieee.org/document/11630600
- Raster2Hull source/data named by the authors: https://github.com/dinceruur/raster2Hull


## 16. G2f-a2: multi-waterplane curves and upright mass-balance receipts

The full-envelope source can now be evaluated at a bounded ordered set of upright horizontal waterplanes without rebuilding the source representation. The implementation is in `src/sectional_reference.rs`, with two separate algorithms:

- `sectional-area-trapezoid-v1`: clips each station's closed transverse profile to the submerged half-plane, computes section area/centroid and waterline breadth, then integrates those piecewise-linear longitudinal curves using compensated summation.
- `upright-mass-balance-bisection-v2`: solves a caller-supplied total mass at explicit water density by safeguarded bracketed bisection over z-down waterplane elevation.

Commands:

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json --output /tmp/marine-sectionals.json

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json --verify /tmp/marine-sectionals.json

    cargo run --locked -p symthaea-marine-geometry --bin marine-upright-mass-balance -- docs/research/marine/fixtures/marine-upright-mass-balance-box-001.json --output /tmp/marine-mass-balance.json

    cargo run --locked -p symthaea-marine-geometry --bin marine-upright-mass-balance -- docs/research/marine/fixtures/marine-upright-mass-balance-box-001.json --verify /tmp/marine-mass-balance.json

The independent verifier recomputes from the exact input request. Mutating a result and recomputing its BLAKE3 receipt digest does not make the changed result valid.

### Known-answer corpus

**Prismatic box hull:** 100 m length, 20 m beam, 6 m design draft, 4 m freeboard, 1,025 kg/m³ explicit density.

| Waterplane z (m, z-down) | Displacement (m³) | Waterplane area (m²) | VCB (m, z-down) |
|---:|---:|---:|---:|
| -2 | 16,000 | 2,000 | 2 |
| 0 | 12,000 | 2,000 | 3 |
| +2 | 8,000 | 2,000 | 4 |

All have LCB = LCF = 50 m, TCB = 0 and TPC = 20.5 t/cm. For increasing immersion (decreasing z), the finite-difference displacement derivative equals the 2,000 m² waterplane area in this exact prismatic fixture.

**Nonuniform, linearly tapered hull:** 100 m length, three stations at x/L = 0, 0.4 and 1.0, half-breadths 5, 7 and 10 m respectively, constant 6 m draft.

- Displacement: 9,000 m³.
- Waterplane area: 1,500 m².
- LCB = LCF = 55.5555555556 m.
- VCB = 3 m, TCB = 0.
- Mass at 1,025 kg/m³: 9,225 t.
- TPC: 15.375 t/cm.

This fixture tests nonuniform station spacing and longitudinal first moments; a constant-area rectangular prism would not expose a misplaced LCB/LCF integration.

**Convergence and invariance controls:** the current tests include a linear station-refinement invariant, a quadratic-breadth second-order convergence test for displacement and waterplane area, density-scaling invariance, explicit asymmetric-port use, above-sheer waterplane refusal, receipt rebinding resistance and upright mass-balance solver stopping/iteration controls.

### Upright mass-balance stop condition correction

The bisection algorithm previously checked the bracket width before incorporating the current midpoint into the bracket. With a small maximum iteration budget this could return iteration-limit failure even though the newly narrowed bracket had already met the caller's plane tolerance. The branch now:
1. evaluates the midpoint;
2. checks the mass residual;
3. updates the appropriate shallow/deep bracket endpoint;
4. checks the **refined** bracket width; and
5. stops with `WaterplaneBracketWithinTolerance` when that refined bracket meets tolerance.

A regression case forces the bracket-width stop in two iterations and verifies that the reported equilibrium remains inside the final bracket. A bracket-width stop does not imply the mass residual also met its separate tolerance; both are recorded and must be assessed distinctly.

### G2f claim ceiling

This reference is independent of the G2c polyhedral integration method but is not a generated displacement mesh or a production hydrostatics package. It assumes piecewise-linear longitudinal interpolation of section area, waterline breadth and sectional first moments. Errors for strongly curved longitudinal variation must be characterized through station-refinement and independent 3D mesh comparisons. Upright sinkage does not solve trim/heel, KG, free surfaces, GZ, stability criteria or flooding.

The next static-attitude work is tracked separately in [MARINE-STATIC-ATTITUDE-EQUILIBRIUM-001 #12498](https://github.com/Luminous-Dynamics/symthaea/issues/12498). It specifies three coupled unknowns (waterplane offset, pitch and roll) with a gravity-aligned force/moment residual; it must not be confused with dynamic WAPS/leeway manoeuvring or with a stability approval.

## 17. Updated competitive comparison

The current market baseline is already strong and should be described precisely:

| System / research | Demonstrated capability | What we should do differently |
|---|---|---|
| NAPA | Approved hydrostatic/compartment models, stability/loading workflows and operational data bridging | Bind as-built/configuration revision, load-case provenance and downstream verification continuously; do not claim an onboard stability replacement |
| Maxsurf Stability | Equilibrium hydrostatics, free/fixed trim, tank/compartment and intact/damage stability workflows | Provide open evidence receipts and independent reconstruction/comparison; do not claim stability parity until benchmarked against the same criteria |
| Siemens STAR-CCM+ | Transient free-surface VOF, dynamic fluid-body interaction and up to 6-DOF vessel motion | Interoperate with specialist CFD and compare per observable under ITTC V&V; do not claim a home-grown model is more accurate |
| SHIPFLOW | Specialist multi-fidelity resistance, propulsion, seakeeping and manoeuvring CFD | Optimize cost to independently validated mission improvement; do not attempt to recreate the full solver suite |
| GAN / VAE-PCT (2026) | 8,000 processed CAD hulls, five quality/diversity metrics, B-spline pull-back/fairing and early calm-water resistance statistics on two single-type datasets | Reproduce at matched data/compute, then add engineering coefficient bias, loading curves, evidence-chain and held-out validation scorecards |
| Representation-aware latent diffusion (2026) | Shared latent encoding across point clouds, waterline sections and buttock curves | Preserve the same multi-representation consistency while making feasibility, identity and independent verification first-class |
| Raster2Hull (2026) | 5,000 synthetic drawing/point-cloud pairs for GA/lines-plan reconstruction | Benchmark real degraded archives separately; require uncertainty and abstention because synthetic point-cloud fit does not establish watertight CAD |

Primary references:
- NAPA loading/stability and digital-twin data bridge: https://www.napa.fi/a-digital-twins-best-foundation-is-meaningful-use-of-data/
- Siemens STAR-CCM+ hull-performance workflow: https://static.sw.cdn.siemens.com/siemens-disw-assets/public/4b92hDfRCo5ND51UfBtk7p/en-US/Siemens-SW-Simcenter-STAR-CCM%2B-Hull-Performance-Workflow-Fact-Sheet_tcm27-80626.pdf
- SHIPFLOW capabilities: https://shipflow.se/
- GAN/VAE-PCT comparison: https://www.mdpi.com/2077-1312/14/18/1707
- Representation-aware generative modeling: https://www.mdpi.com/2077-1312/14/7/642
- Raster2Hull: https://xplorestaging.ieee.org/document/11630600
- ITTC 2024 CFD benchmark database, 7.5-03-02-02 Rev.03: https://ittc.info/media/11956/75-03-02-02.pdf
- ITTC 2024 CFD uncertainty examples, 7.5-03-02-01 Rev.02: https://www.ittc.info/media/11954/75-03-02-01.pdf

### Falsifiable differentiation targets

Do not compare headline scores from unrelated papers. Reproduce the appropriate baselines with frozen corpora, parent-hull-family-held-out splits, normalization, test conditions and compute/evaluation budgets.

Our research target is to demonstrate all of the following together:
- 100% of admitted full-envelope region and boundary accounting is reconciled;
- 0 silent geometry repair, frame/unit inference or hidden conversion;
- multi-waterplane displacement curves converge under station refinement with documented order and per-observable residuals;
- sectional and polyhedral methods compare within predeclared combined numerical uncertainty on matched geometry;
- independent CAD round-trip retains geometry, persistent identity and validation properties;
- resistance/propulsion/seakeeping outputs are only marked validated when comparison error is within independently estimated combined validation uncertainty for that exact observable/condition;
- held-out model predictions are calibrated by hull family and operating regime, with false-feasible and rank-reversal rates reported;
- final system wins on independently confirmed Pareto improvement per fixed specialist-evaluation budget and lifecycle engineering cost, not raw number of generated candidates.

These remain research targets, not achieved performance.


## 12. Independent multi-waterline sectional reference (G2f-d research bridge)

The current branch also adds `sectional_reference.rs`, a deliberately separate numerical route for evaluating multiple upright horizontal waterplanes from the same full-envelope input. It is not a mesh generator. It is intended to become a second hydrostatic comparison against G2c polyhedral integration once G2f-b/c surface generation is available.

### Interpolation contract

For each station, the method builds the full transverse section perimeter from the starboard profile and either:
- an exact reflected port profile under `SYMMETRIC_ABOUT_Y0`, or
- the explicitly supplied port profile under `EXPLICIT_PORT_AND_STARBOARD`.

It clips this section to the submerged half-plane (z \ge z_w), with z-down coordinates. It then computes area, transverse/vertical section centroid and waterline breadth.

Between longitudinal stations, the v1 reference declares piecewise-linear interpolation of section area, waterplane breadth and section centroid. The longitudinal integration uses:
- trapezoidal integration of area for displacement volume;
- trapezoidal integration of waterline breadth for waterplane area;
- exact integration of (x f(x)) when (f(x)) is linearly interpolated for LCB/LCF first moments;
- exact integration of (f(x)g(x)) when both fields are linearly interpolated for transverse and vertical area moments.

The last point is important: using a trapezoid on endpoint products (A_i\bar z_i) or (A_i\bar y_i) is not exact because the product of two linearly varying fields is quadratic. The reference now uses the closed-form integral of that quadratic product.

This is still an approximation model for the longitudinal behavior between the supplied stations. It must not be represented as exact integration of an arbitrary CAD loft. Refinement and convergence studies are required before using it as a decision-grade reference on real vessels.

### Synthetic loading-sweep fixture

The box fixture carries three horizontal waterplanes:
- z=-2 m (deeper/overload waterline): 16,000 m³;
- z=0 m (design waterline): 12,000 m³;
- z=+2 m (shallower/light waterline): 8,000 m³.

At explicit density 1,025 kg/m³, all three have 2,000 m² waterplane area and TPC 20.5 tonnes/cm. LCB and LCF are 50 m. VCB is 2, 3 and 4 m respectively in the declared z-down coordinate frame. These are analytical synthetic fixture values.

A second nonuniform-depth test varies the station draft from 4 m to 8 m across a 100 m length. Under the v1 piecewise-linear section-observable model it has exactly:
- displacement 12,000 m³;
- LCB (500/9\) m;
- VCB (28/9\) m.

This detects the bias from trapezoidal integration of area-centroid endpoint products.

### Receipt and claim semantics

Each result binds:
- the full-envelope source specification digest;
- a unique loading-condition ID;
- explicit horizontal waterplane elevation;
- explicit water density;
- the algorithm identity and calculated observables.

The ordered sweep digest binds the complete result list. Receipt verification replays the same exact request; it verifies determinism and integrity, not independent implementation diversity. The independent numerical-method claim is specifically that sectional clipping/integration is separate from G2c mesh tetrahedral integration. The two methods still need to be run on the same generated geometry before claiming cross-method agreement.

The elevation is supplied by the caller; this module does not solve a vessel's draft/trim equilibrium from mass, tank contents or KG. It supports neither trim nor heel. It refuses waterplanes above the supported sheer section rather than creating unmodelled deck flooding.

CLI:

    cargo run --locked -p symthaea-marine-geometry --bin marine-sectional-hydrostatics-reference -- docs/research/marine/fixtures/marine-full-envelope-loading-sweep-001.json

Exit code 0 reports an evaluated receipt, 1 reports typed validation/integration rejection and 2 indicates CLI, I/O or JSON parsing failure.



## 13. G2f-b first full-envelope surface-loft implementation

Branch: `research/marine-full-envelope-mesh-g2f-b-20261117`.

The first G2f-b source slice adds `full_envelope_mesh.rs`, a deterministic triangle-mesh generator distinct from the G2b displacement-volume loft. It generates:
- port/starboard bottom and topside strips from compatible station profile segments;
- the deck as a separately identified lofted surface;
- transom end caps via a deterministic fan only when the complete cross-section perimeter is convex;
- no planar cap at a pointed stem; side and deck faces converge on declared shared centreline identities.

### Identity and topology

Every vertex is bound to a station-scoped source point identity. Symmetric-derived port points use `MIRROR:<source-id>`; centreline points reuse the same identity. Explicit asymmetric profiles use their supplied port IDs, with a shared keel identity required for the closed v1 shell. Deck-edge vertices reuse exact sheer identities. Coordinates that happen to coincide do not cause welding.

The generator uses the G2e-compatible stern-interval diagonal flip when a stern collapses to the centreline. It preflights vertex/face budgets with checked arithmetic and rejects unsupported cross-station semantic mismatches, unclosed transom identities, non-convex cap polygons and opening loops. It does not repair or fill geometry.

The output `FullEnvelopeMeshArtifact` binds:
- source schema and full-envelope spec digest;
- station-scoped source-point/vertex identity digest;
- ordered Bottom, Topsides, Deck, BowEndClosure and SternEndClosure face ranges;
- exact triangle mesh digest;
- caller-supplied G0/G1 admission profile and fresh admission report;
- full-envelope mesh manifest digest.

The verifier regenerates from the exact source spec and admission profile and compares the complete artifact. A caller cannot mutate face indices, recompute the mesh/manifest hashes and obtain a valid replay.

### Synthetic known answers and tests authored

The symmetric box hull has full external volume 20,500 m³, including its deck crown. This is distinct from the 12,000 m³ z=0 displacement shell.

The current Rust unit suite covers transom/transom, pointed-bow/transom, transom/pointed-stern, pointed/pointed with an interior full-breadth station, explicit asymmetric geometry, non-convex transom rejection, profile-count mismatch, shared asymmetric keel identity, opening-loop incompleteness, semantic face-range coverage, deterministic replay, rehashed face mutation and unknown G0/G1 checks. A CLI plus integration tests exercise generation and verification.

### Competitive position and benchmark refinement

The 2026 representation-aware hull-generation paper already combines surface point clouds, waterline splines and buttock curves in a shared latent space with conditional latent diffusion. Its dataset is 15,000 synthetic hulls produced by 625 variations of each of 24 parent hulls; the authors explicitly note that finite parent-hull diversity limits generalization and that the current approach assumes symmetric, watertight monohulls. The correct comparison is not to claim multimodal generation as new, but to benchmark whether explicit source identities, asymmetric geometry and replayable topology improve out-of-family and damaged/legacy reconstruction.

The Raster2Hull 2026 benchmark introduces 5,000 synthetic pairs of rasterized general-arrangement/lines drawings and hull point clouds, while explicitly acknowledging that real degraded scans and watertight CAD reconstruction remain unsolved. Our reconstruction benchmark should therefore report drawing-to-surface error, unsupported/ambiguous geometry, cross-view consistency and CAD recoverability separately; synthetic raster performance is not real-fleet accuracy.

References:
- Htein et al. (2026), *Representation-Aware Generative Modeling for Early-Stage Ship Design*: https://www.mdpi.com/2077-1312/14/7/642
- Uur et al. (2026), *Raster2Hull*: https://doi.org/10.1109/ACCESS.2026.3717961
- Liu et al. (15 Nov 2026), *An integrated data-driven and experimental methodology for toroidal propeller design improvement*: https://doi.org/10.1016/j.engappai.2026.116126

### Next engineering benchmark

The most valuable near-term benchmark is cross-representation recovery on the *same immutable design revision*:
1. Generate the external mesh from station profiles/deck sections.
2. Independently slice it into transverse sections, waterlines, buttock curves and surface point clouds.
3. Reconstruct the same source representations from those slices.
4. Compare point/surface Hausdorff and percentile distances, curve deviations, integral hydrostatic properties, boundary/seam identity and AP242 persistent IDs.
5. Repeat across transom, pointed, asymmetric and held-out parent families.

Targets are research aspirations, not achievements: zero unaccounted surface regions; zero boundary/non-manifold edges for admitted closed shells; zero silent geometry repair; 100% exact source-to-vertex identity recovery; 100% manifest replay; and bounded surface/curve/property residuals under frozen tolerances. Only after independent CAD-kernel round-trip should we claim CAD recovery; only after specialist CFD/physical comparison should we claim hydrodynamic improvement.
