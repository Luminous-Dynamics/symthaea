# Mars Tether — Next Computational Experiment

## Objective

Implement the smallest falsifiable T0/T1 model before introducing full flexible-tether FEM.

## New design rule

Treat material feasibility as a first-class uncertainty parameter, not a fixed material choice. Published reviews still identify ultra-high specific strength and scalable manufacturing as central constraints; candidate graphene/graphitic laminates have promising laboratory properties but important large-scale and interface properties remain unresolved.

## T0 outputs

- Mars body-fixed anchor position;
- nominal areosynchronous radius;
- local rotating-frame geometry;
- tether azimuth/elevation family;
- endpoint state;
- surface/topographic intersection checks;
- Phobos/Deimos radial-region crossing diagnostics.

## T1 outputs

- distributed mass model;
- taper parameterization;
- counterweight parameterization;
- center-of-mass state;
- static tension envelope;
- required working stress;
- specific-strength requirement;
- sensitivity to safety factor and manufacturing defect assumptions.

## Parameter sweep

Do not select a material first. Sweep:

- density;
- ultimate tensile strength;
- allowable working stress;
- defect/degradation factor;
- taper law;
- payload mass;
- climber mass;
- counterweight mass;
- anchor elevation;
- anchor latitude;
- tether azimuth.

Convert material inputs to specific strength and report the required envelope.

## Phobos/Deimos treatment

Phobos is not merely an avoidance object. Its mean orbital distance is about 9,378 km from Mars' center, with 1.08 deg inclination and 0.0151 eccentricity. Deimos is at about 23,459 km, with 1.79 deg inclination and 0.0005 eccentricity.

Therefore the solver should calculate time/phase-dependent clearance and not rely on a static radial-distance test.

## Evidence states

Every result must carry:

- model revision;
- constants source/version;
- ephemeris source/version;
- solver revision;
- parameter set;
- numerical tolerances;
- uncertainty bounds;
- execution mode;
- evidence status.

Dry-run or fixture results must never be promoted to solver-backed evidence.

## Why this ordering matters

The literature continues to support the general Mars-elevator concept as a serious theoretical engineering problem, but material strength, manufacturing scale, climber/tether interface properties, and dynamics remain major unresolved issues. A useful Symthaea implementation should expose those constraints quantitatively instead of hiding them behind a nominal design.

## Exit criteria for T0/T1

The experiment is ready to advance to flexible dynamics only when:

1. reference-frame conversions are deterministic and tested;
2. the rotating geometry closes numerically;
3. mass/taper inputs reproduce known limiting cases;
4. tension and specific-strength calculations have independent checks;
5. Phobos/Deimos clearance is phase-aware;
6. uncertainty is preserved in machine-readable output;
7. no site ranking is produced;
8. every numerical result is provenance-bearing.
## First-order numerical sanity check

Using Mars GM = 4.2828372e13 m^3/s^2, equatorial radius = 3,396.2 km, and sidereal period = 88,642.44 s gives:

- rotation rate ≈ 7.088236e-5 rad/s;
- areosynchronous radius ≈ 20,427.65 km from Mars center;
- areosynchronous altitude above the equatorial radius ≈ 17,031.45 km;
- effective radial acceleration at the surface ≈ -3.696 m/s^2 in the rotating frame;
- effective radial acceleration at Phobos' mean radius ≈ -0.440 m/s^2;
- effective radial acceleration at areosynchronous radius ≈ 0;
- effective radial acceleration at Deimos' mean radius ≈ +0.040 m/s^2.

For a constant-density, constant-allowable-stress idealized tether segment, the first-order self-supporting area-ratio exponent from radius r0 to r1 is:

    I(r0,r1) = GM(1/r0 - 1/r1) - 0.5*omega^2*(r1^2-r0^2)

and A(r1)/A(r0) = exp(I / specific_strength), where specific_strength = allowable_stress / density.

For the surface-to-areosynchronous interval, I ≈ 9.495e6 m^2/s^2. The corresponding idealized area ratios are approximately:

- 5 MYuri: 6.68x;
- 10 MYuri: 2.58x;
- 15 MYuri: 1.88x;
- 30 MYuri: 1.37x;
- 50 MYuri: 1.21x.

These are sanity-check numbers only. They do not establish a feasible elevator because they omit the full counterweight/apex boundary condition, climber loads, local topography, bending, defects, fatigue, thermal state, dynamic stability, deployment architecture, and perturbations. Their value is that they give the T1 implementation an independently checkable analytical limiting case.

## Architectural consequence

The first model should therefore report specific-strength requirement curves, not merely a yes/no material result. This lets material manufacturing evidence enter later as an uncertainty distribution and allows the same tether architecture to be evaluated against future improvements in graphene, CNT, hBN, or other candidate materials.

Recent literature still identifies ultra-high specific strength, long-length manufacturing, and climber/tether interface properties as central unresolved engineering constraints.

## T0 reference kernel now implemented

The first auditable mechanics kernel now lives in `crates/domains/symthaea-physics/src/mars_tether.rs`. It intentionally contains no ephemeris or flexible-tether assumptions.

It provides:

- Mars GM, radius, rotation period, and Phobos/Deimos reference radii;
- synchronous radius and altitude;
- rotating-frame radial effective acceleration;
- the analytic taper integral;
- constant-stress area-ratio calculation;
- Simpson quadrature for tether mass per unit anchor area;
- an explicit diagnostic for whether a radial tether interval crosses a satellite's mean orbit;
- regression tests for the analytical limiting cases.

The mass model is deliberately expressed as **mass per unit anchor area** rather than total tether mass. That keeps the first experiment honest: without a specified anchor cross-section, counterweight, termination condition, payload schedule, and structural architecture, a total mass claim would imply more certainty than the model contains.

## Ephemeris boundary

The next orbital-coupling step should consume authoritative SPICE-derived ephemerides rather than embedding a hand-built Phobos/Deimos propagator in this module. NASA NAIF describes SPICE as an engineering-grade geometry system and its Martian archives include SPK ephemerides for Mars, Phobos, Deimos, and the Sun.

Therefore:

1. `mars_tether` remains the deterministic analytic reference layer;
2. an orbital/ephemeris adapter supplies time-tagged body states;
3. the tether solver consumes those states through an explicit provenance-bearing interface;
4. phase-aware clearance becomes a downstream T2/T3 computation;
5. external structural/dynamic solvers can later feed validated loads back into the same evidence boundary.

This separation prevents a convenient local propagator from silently becoming an authoritative celestial-mechanics source.

## Literature-derived refinement

The recent Mars-Phobos elevator literature also reinforces the need to keep the architectures separate. A 2025 Acta Astronautica study models a Phobos-to-Mars tether around the Mars-Phobos L1 region using a planar elliptic restricted three-body problem. That is a different boundary-value problem from a surface-to-areosynchronous elevator and should be represented as a sibling architecture, not folded into the T0 surface solver.

Material work likewise remains parameter-driven: reviews identify ultra-high specific strength and scalable production as central requirements, while climber-interface work highlights unresolved friction, shear, thermal, and anisotropy measurements. Those unknowns should remain explicit uncertainty fields rather than being replaced by a nominal graphene/GSL value.


## T0.5 geometry kernel now implemented

The analytic layer now includes a spherical Mars anchor and straight-line rotating tether geometry:

`MarsAnchor -> ENU basis -> tether direction -> endpoint -> corotation velocity`.

The coordinate contract is explicit:

- latitude is areocentric;
- longitude is positive east;
- anchor elevation is above the reference spherical Mars radius;
- azimuth is measured east of north;
- elevation is measured above the local horizontal plane.

A quadratic sphere-intersection diagnostic determines where a straight tether first reaches a specified Mars-centered radius. This gives us a clean bridge to the areosynchronous sphere without pretending that the tether is already a solved flexible structure.

This is important for non-equatorial designs. Published non-equatorial elevator analysis explicitly models latitude-dependent taper/payload effects rather than treating an off-equator anchor as a simple translation of the equatorial solution. In particular, the literature reports reduced payload capacity with increasing anchorage latitude and increased deployment latitude range with higher tensile strength.

## Topography boundary is now clearer

The surface geometry should not query a generic "Valles altitude" value. It should consume a terrain provider returning elevation plus uncertainty and provenance at the requested latitude/longitude.

MOLA-derived Mars topography provides a suitable initial reference source. USGS describes a MOLA-based global DEM and reports approximately 100 m horizontal-position accuracy and about 1 m radial accuracy for the underlying points, while also documenting interpolation gaps and areoid uncertainty. NASA's current open-data catalog continues to expose the MOLA mission gridded records.

For Valles specifically, the USGS geologic map distinguishes the Noctis Labyrinthus plateau, the Valles Marineris province, and the eastern canyon province, reinforcing the architectural rule that "Valles Marineris" is a region containing materially different terrain and geology rather than one anchor site.

The next terrain interface should therefore return something like:

`TerrainSample { elevation_m, elevation_uncertainty_m, slope_rad, roughness, geology_class, source_id, source_revision }`.

No site-ranking function should consume this directly. It should feed constraint predicates and uncertainty bounds.

## Ephemeris boundary confirmed

JPL's planetary-satellite ephemeris service publishes SPK files intended for use with the NAIF SPICE toolkit, including Martian satellite ephemerides. NAIF explicitly states that SPICE is also used for engineering tasks.

This makes the intended T2/T3 interface:

`EphemerisProvider(time, frame) -> BodyState(position, velocity, provenance)`.

The tether kernel should remain agnostic about the particular SPICE kernel revision. Kernel identity and coverage interval become evidence fields.

## Architectural consequence

We now have three clean fidelity boundaries:

1. **T0:** analytic Mars reference mechanics;
2. **T0.5:** anchor/local-frame/straight-tether geometry;
3. **T1+:** distributed mass, phase-aware ephemerides, flexible dynamics, environmental loads and operations.

That ordering lets us test each layer independently instead of allowing a sophisticated simulator to conceal a coordinate or reference-frame error.


## Terrain evidence contract and first site-assessment state machine

The Rust kernel now defines `TerrainSample`, `TerrainProvenance`, `TerrainQuality`, a `TerrainProvider` adapter boundary, and `AnchorGeometryAssessment`. A geometry assessment with a usable terrain sample is intentionally marked `HigherFidelityRequired`, not certified feasible: terrain/geology, structural equilibrium, and tether dynamics are not solved by the spherical T0 geometry.

The initial terrain adapter should use the MOLA MEGDR/DEM coordinate convention explicitly: areocentric (planetocentric) latitude and east-positive longitude. The USGS global MOLA DEM is commonly distributed at 128 pixels per degree (about 463 m/pixel at the equator); gaps between tracks are common and some grid values are interpolated. USGS reports total elevation uncertainty of at least about ±3 m for the cited product due to areoid and regional-shape uncertainty. These values describe that dataset/product, not a universal guarantee for all MOLA derivatives. The PDS archive also distinguishes planetary radius, areoid, topography, and observation count, so adapters must preserve the vertical datum and whether a sample is measured or interpolated.

Consequently, the present kernel does **not** silently turn MOLA areoid-relative elevation into spherical radius. A future adapter must explicitly transform the selected terrain datum into the geometry model's radius convention and carry the transformation revision and uncertainty. Until then, a terrain sample is evidence that a location was queried—not proof that the structural anchor elevation is known.

## Scope and verification status

The repository change contains unit tests authored for geometry and evidence-state behavior, but this workflow has not run the Rust workspace test suite or compiler. Treat the implementation as committed source awaiting CI/compiler verification; do not describe the tests as passing until CI confirms them. No MOLA raster was ingested and no Valles latitude/longitude sweep has been executed in this iteration.


## Datum conversion and sample-to-anchor identity

The kernel now includes an explicit `radial_elevation_from_areoid` conversion. It requires an areoid radius already expressed in a compatible body-fixed frame and returns radial elevation relative to the kernel's spherical reference radius; it does not infer an areoid from topography. Terrain assessments also reject samples whose latitude/longitude do not match the requested anchor within the declared numerical tolerance.

The PDS MEGDR archive distinguishes planetary radius, areoid, topography, and observation counts, and defines the current MEGDR coordinate system as areocentric latitude with east-increasing longitude. Its maintained PDS4 bundle includes labels and ENVI headers alongside image files. This supports a future adapter that can preserve product identity and datum metadata instead of treating a bare raster value as self-describing. The adapter must still select a compatible areoid/radius field and coordinate convention; this contract alone does not establish a geodetic transformation or geophysical validity.

A MOLA-backed geographic sweep remains blocked on implementing and validating the raster adapter and acquiring a specific, versioned product. No candidate site should be labelled feasible from the present spherical kernel.


## Input validation hardening

The anchor assessment now rejects a non-finite or non-positive minimum anchor radius as `InsufficientEvidence` before using the configuration to classify a candidate. This is a configuration-quality failure, not evidence that a geographic site is physically infeasible. A regression test covers a NaN threshold. The assessment remains intentionally non-certifying: a usable terrain sample is not yet datum-converted and compared against the structural anchor elevation.


## Explicit terrain vertical datum and uncertainty propagation

The terrain contract now carries a `TerrainVerticalDatum` for each sample: areoid-relative height, height above the kernel reference sphere, or planetocentric radius. The new `radial_terrain_elevation` conversion returns reference-sphere-relative height and an uncertainty bound. For areoid-relative samples, both the compatible areoid radius and its uncertainty are mandatory; the converter combines the areoid and sample uncertainty bounds by addition (worst case), not root-sum-square, because independence is not established. Missing or invalid datum inputs return no conversion rather than a guessed height.

This makes the PDS MEGDR product distinction actionable in code: adapters must declare the exact vertical quantity represented by the raster and provide a compatible areoid model when required. Conversion does not itself establish geodetic compatibility, terrain accuracy, or anchor suitability. A geographic sweep remains gated on a real, versioned raster adapter and product-level validation.


### Terrain sample admissibility

The core sample validator also checks that areocentric latitude lies within [-90°, +90°], optional slope lies within [0°, 90°], and provenance identifiers (source, revision, coordinate reference) are non-empty. These are syntactic admissibility checks only; they do not prove that a product label is truthful, that the grid registration is correct, or that a coordinate transform is valid. Dataset adapters remain responsible for validating those claims against source metadata.


## MOLA raster adapter: projection and record-layout hardening

The MOLA adapter now treats the projection metadata as part of the georeferencing contract rather than incidental label text. In addition to SIMPLE CYLINDRICAL, planetocentric latitude, east-positive longitude, and explicit line/sample projection offsets, the label must declare `COORDINATE_SYSTEM_TYPE = "BODY-FIXED ROTATING"` and `MAP_PROJECTION_ROTATION = 0.0`, matching the authoritative MOLA EGDR SIS global example. Companion topography/count products must match the projection center and offsets as well as grid dimensions and geographic bounds.

The raster reader also validates that the declared sample payload fits inside each fixed-length record. Sixteen-bit count decoding follows the declared integer byte order instead of assuming big-endian encoding. These checks are deliberately fail-closed: a malformed label or inconsistent companion cannot silently shift registration or reinterpret bytes.

The remaining high-value gate is **product-byte evidence**, not another inferred projection formula. NASA/PDS documents the MOLA MEGDR 128-ppd tiled archive and the 00N270 tile containing the Valles Marineris longitude band, but the exact Valles label bytes still need to be pinned and hashed before changing the current longitude transform. Generic PDS projection equations and other planetary archives exhibit historical offset/sign convention differences, so the adapter should not substitute a generic equation for the actual MOLA product convention without an exact label-backed regression. Once the authoritative label and image/count bytes are available, the next evidence gate is: content hashes -> exact metadata snapshot -> known-cell byte fixtures -> topography/count registration tests -> coordinate sweep across tile boundaries.

## MOLA tile-footprint gate

The raster adapter now rejects coordinate queries outside the geographic footprint declared by a tiled MEGDR label before projection arithmetic is applied. This is deliberately separate from the unresolved MOLA longitude-transform question: a tile-local reader must never wrap a longitude from another tile into an apparently valid cell. The 128 ppd archive explicitly partitions the equatorial product into 90-degree longitude tiles, including the 270E–360E tile containing Valles Marineris. NASA/PDS documentation also establishes that the maintained archive contains the original PDS3 labels alongside the PDS4 products. citeturn5view0turn3search9

A second evidence boundary is now explicit: published downstream MOLA processing reports that LINE_PROJECTION_OFFSET and SAMPLE_PROJECTION_OFFSET are off by one pixel in several MEGDR label files and were corrected in derived processing. That is strong evidence that projection offsets cannot be treated as self-validating merely because they parse, but it does not identify the correction for the Valles tile itself. The adapter therefore remains conservative: exact Valles label bytes and image/count bytes must still be pinned before changing the current longitude transform. citeturn2search0


### MOLA archive provenance gate (2026-09-30)

The PDS Geosciences archive now exposes the migrated PDS4 MEGDR bundle and the legacy PDS3 products for the 128-pixels/degree collection. The archive identifies the Valles-containing megt00n270hb.img as a 129,761,280-byte product and its companion PDS3 label as 4,822 bytes. The NASA PDS collection identifies the migrated 128-ppd collection as urn:nasa:pds:mgs_mola_topography_derived:data_meg128::1.0, under bundle DOI 10.17189/1z1b-kv84.

This establishes a stronger archive identity, but it is not a substitute for content hashing. The adapter must not record a SHA-256 value until the exact binary and label bytes have been retrieved and hashed from the archive. The current execution environment cannot resolve the PDS archive host, so no hash is asserted here. The next provenance fixture should capture, at minimum, the exact PDS4 product identifier, PDS3 label byte hash, IMG byte hash, file sizes, retrieval timestamp, and the known-cell byte/value fixtures for both MEGT00N270HB and MEGC00N270HB.
