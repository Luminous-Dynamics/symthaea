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

This is important for non-equatorial designs. Published non-equatorial elevator analysis explicitly models latitude-dependent taper/payload effects rather than treating an off-equator anchor as a simple translation of the equatorial solution. In particular, the literature reports reduced payload capacity with increasing anchorage latitude and increased deployment latitude range with higher tensile strength. citeturn0search5

## Topography boundary is now clearer

The surface geometry should not query a generic "Valles altitude" value. It should consume a terrain provider returning elevation plus uncertainty and provenance at the requested latitude/longitude.

MOLA-derived Mars topography provides a suitable initial reference source. USGS describes a MOLA-based global DEM and reports approximately 100 m horizontal-position accuracy and about 1 m radial accuracy for the underlying points, while also documenting interpolation gaps and areoid uncertainty. citeturn0search2 NASA's current open-data catalog continues to expose the MOLA mission gridded records. citeturn0search4

For Valles specifically, the USGS geologic map distinguishes the Noctis Labyrinthus plateau, the Valles Marineris province, and the eastern canyon province, reinforcing the architectural rule that "Valles Marineris" is a region containing materially different terrain and geology rather than one anchor site. citeturn0search1

The next terrain interface should therefore return something like:

`TerrainSample { elevation_m, elevation_uncertainty_m, slope_rad, roughness, geology_class, source_id, source_revision }`.

No site-ranking function should consume this directly. It should feed constraint predicates and uncertainty bounds.

## Ephemeris boundary confirmed

JPL's planetary-satellite ephemeris service publishes SPK files intended for use with the NAIF SPICE toolkit, including Martian satellite ephemerides. citeturn0search6 NAIF explicitly states that SPICE is also used for engineering tasks. citeturn0search0

This makes the intended T2/T3 interface:

`EphemerisProvider(time, frame) -> BodyState(position, velocity, provenance)`.

The tether kernel should remain agnostic about the particular SPICE kernel revision. Kernel identity and coverage interval become evidence fields.

## Architectural consequence

We now have three clean fidelity boundaries:

1. **T0:** analytic Mars reference mechanics;
2. **T0.5:** anchor/local-frame/straight-tether geometry;
3. **T1+:** distributed mass, phase-aware ephemerides, flexible dynamics, environmental loads and operations.

That ordering lets us test each layer independently instead of allowing a sophisticated simulator to conceal a coordinate or reference-frame error.
