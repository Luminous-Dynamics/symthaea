# LL-003D Synthetic Terrain / Protected-Zone Oracle — Evidence Note

Status: independent geometry oracle for exact fixtures. Not a real lunar corridor model and not release authority.

## Purpose

Before consuming LOLA topography or any mission-specific protected-site registry, the corridor layer needs exact synthetic cases whose correct answer is known in advance. This oracle tests only terrain interpolation, conservative clearance accounting, and protected-volume intersection.

## Model

Inputs:

- monotonic 1-D terrain profile: downrange coordinate, elevation, elevation uncertainty;
- sampled 3-D trajectory: downrange/crossrange/height plus trajectory-position uncertainty proxy;
- optional axis-aligned protected volumes;
- explicit uncertainty multiplier.

The oracle linearly interpolates terrain/elevation uncertainty at every trajectory sample and reports:

- minimum nominal clearance;
- minimum conservative clearance;
- downrange coordinate of the controlling sample;
- protected-zone intersection state and intersected IDs.

Conservative clearance is the research fixture relation:

`clearance_lower = trajectory_height - terrain_height - k * (terrain_sigma + trajectory_sigma)`

This deliberately simple additive uncertainty law is an oracle fixture, not a final statistical safety model.

## Executed fixtures

The self-test was executed independently before commit and passed:

1. triangular terrain profile with exact `20 m` nominal minimum clearance;
2. `k=3` conservative minimum of `12.5 m` for the declared synthetic uncertainties;
3. increasing trajectory uncertainty strictly decreases conservative clearance;
4. a trajectory sample outside terrain coverage raises an error rather than assuming zero elevation;
5. deterministic intersection with a synthetic habitat volume;
6. nonintersection with a volume above the trajectory;
7. boundary-touch counts as intersection.

## Protected-zone semantics

The synthetic oracle uses axis-aligned 3-D boxes and an exact segment/AABB slab test. This is only an exact geometry fixture. Production lunar policy may use geodesic polygons, volumes, altitude bands, time-dependent corridors, or other representations; those must preserve the same fail-closed boundary behavior.

## Non-claims

This artifact does not:

- ingest LOLA;
- define a lunar datum/frame transform;
- model terrain between trajectory samples;
- model GRAIL gravity;
- assign acceptable crew/public risk;
- define clearance margins;
- define protected lunar sites;
- classify a route as safe;
- authorize launch.

## Promotion path

1. production terrain/protected-zone implementation agrees with these synthetic fixtures;
2. provenance-bound LOLA adapter with explicit datum/frame/resolution/uncertainty;
3. trajectory sampling/refinement sufficient to avoid missing terrain peaks;
4. LL-005 uncertainty corridor integration;
5. safe-miss consequence classes;
6. deterministic LL-008 admission inputs;
7. mission-specific qualification/governance evidence.

Tracks #1572.
