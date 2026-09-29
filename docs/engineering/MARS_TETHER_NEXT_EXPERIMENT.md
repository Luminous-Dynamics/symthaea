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