# LL-009K — Multi-resolution conservative lunar horizon core

## Purpose

LL-009K reduces already-materialized, provenance-bound lunar terrain samples into one conservative horizon envelope per azimuth bin for LL-009J.

It deliberately does **not** own GeoTIFF/COG access. Raster materialization is a separate offline evidence adapter so GIS dependencies and range-window choices cannot silently change horizon semantics.

## Inputs

Each study supplies:

- one Moon-centered candidate-site position and declared pole vector;
- common `study_id`, `frame_contract_id`, `epoch_contract_id`, and `site_id`;
- one or more terrain layers;
- exact source/materialization SHA-256 values;
- layer radial range `[min_range_m, max_range_m]`;
- native/source-frame lineage and any frame-bridge receipt;
- terrain samples as physical 3-D lunar vectors with vertical uncertainty;
- candidate-site vertical uncertainty;
- explicit azimuth bin width dividing 360 degrees exactly.

Every terrain sample must fall inside its source layer's declared radial range. A coarse far-field layer cannot silently substitute for a required local layer, and vice versa.

## Geometry

For site vector `s` and terrain vector `p`, let

- `u = s / |s|` be local up;
- `d = p - s` be the site-to-terrain vector;
- `d_up = dot(d, u)`;
- `d_horizontal = |d - d_up u|`.

The geometric elevation angle is

`atan2(d_up, d_horizontal)`.

This keeps lunar curvature in the Moon-centered 3-D geometry instead of approximating the Moon as a flat plane over long ranges.

The local north/east convention is deterministic and follows the same polar-basis rules used by LL-009J.

## Conservative elevation

Each admitted sample contributes a conservative elevation upper bound derived from:

- nominal 3-D geometry;
- terrain vertical uncertainty;
- candidate-site vertical uncertainty;
- declared additional angular/frame/geolocation margin where still unresolved upstream.

Increasing any admitted uncertainty term must never reduce the sample's conservative elevation or the final bin envelope.

## Multi-resolution envelope

For every azimuth bin, LL-009K evaluates all admitted samples from all declared local and far-field terrain layers and retains the maximum conservative elevation.

The winning record includes:

- conservative and nominal elevation;
- azimuth/range;
- source layer ID;
- source artifact/materialization hash;
- sample ID/location;
- uncertainty terms;
- frame-bridge lineage;
- optional runner-up provenance for sensitivity analysis.

A farther/coarser layer is allowed to win when it contains the physically higher skyline obstruction. Resolution priority never overrides geometry.

## Determinism

Input ordering cannot change the envelope. Ties use deterministic provenance-based ordering and are recorded rather than depending on iteration order.

The output pack is canonical and hashable. A differing existing immutable output is not overwritten.

## Fail-closed rules

Promoted evidence rejects:

- study/frame/epoch/site lineage mismatch;
- non-finite or degenerate geometry;
- terrain sample located at the site;
- undeclared source or source-hash drift;
- sample outside its layer radial range;
- invalid/overlapping nondeterministic layer policy;
- azimuth bin width that does not divide 360 degrees exactly;
- missing required azimuth bins;
- missing uncertainty semantics;
- silent DE421/DE440 relabeling;
- missing winning-source provenance.

Missing terrain is not interpreted as a zero-degree horizon.

## Executed synthetic gates

The dependency-free self-test covers:

1. closed-form spherical elevation-angle checks;
2. order-independent maximum-envelope selection;
3. a farther/coarser layer beating the local layer when it contains the higher obstruction;
4. monotonicity under larger site/terrain uncertainty;
5. rejection of samples outside declared radial bands;
6. rejection of missing azimuth coverage.

These are synthetic mechanics checks only.

## Handoff to LL-009J

LL-009J consumes the K horizon pack as its promoted terrain-horizon source. J then compares time-sampled Sun/Earth/relay directions against that conservative envelope and derives illumination/LOS intervals.

The intended real-data chain is:

`NASA PGDA/LOLA native rasters -> offline COG/window materializer -> normalized terrain sample pack -> LL-009K -> LL-009J -> LL-009I`.

## Non-claims

A passing LL-009K artifact does not establish:

- site suitability;
- terrain completeness beyond its declared source/range contract;
- delivered solar power;
- RF link performance;
- thermal suitability;
- construction qualification;
- corridor safety;
- launcher/elevator superiority;
- physical release authority.

It is a Phase-0 geometric evidence reducer only.
