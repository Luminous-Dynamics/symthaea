# LL-009P — Site01 reference-anchor materialization

## Purpose

LL-009P removes the final manually entered site-state fields from the first Site01 terrain lineage.

NASA's Lunar Surface Data Book identifies `R01 / Site01 / Connecting Ridge` at latitude `-89.4632 deg`, longitude `-137.49 deg`. LL-009P treats this as a **published reference anchor**, not as a claim that this point is the optimal or selected infrastructure location.

The anchor is resolved against exact LL-009N-locked Site01 elevation and total-Z-uncertainty rasters. The containing source pixel is selected without interpolation, and its center, elevation and uncertainty become an immutable LL-009L-ready site block.

## Why pixel-center materialization matters

A tempting implementation would keep the exact published latitude/longitude as X/Y while borrowing Z and uncertainty from whichever raster cell happens to contain it. That creates a hybrid state whose horizontal position and terrain support point differ by up to several meters.

V1 instead records both:

- the exact published reference and its projected X/Y;
- the exact containing source cell and pixel-center X/Y/lon/lat.

The LL-009L site block uses the pixel center. The receipt reports the reference-to-center offset and fails if it exceeds the configured limit.

For the 5 m Site01 grid, the contract permits at most `4 m`; the theoretical half-diagonal of an unrotated 5 m square pixel is about `3.54 m`.

## Exact source binding

The input is an `ll009n.nasa-source-lock.v1` receipt. LL-009P independently re-hashes the exact elevation and uncertainty files and, when present, rechecks the locked byte sizes before opening either raster.

The accepted source roles are deliberately narrow:

- elevation: `surface_elevation`;
- uncertainty: a recognized vertical-error role such as `elevation_rms_uncertainty_m`.

The Site01 config binds:

- `site01-elevation-5m`;
- `site01-total-z-rms-5m`.

No filename-only lookup is sufficient.

## Raster and projection contract

Elevation and uncertainty must have exactly matching:

- width and height;
- CRS;
- affine transform;
- single-band layout;
- expected pixel scale.

The observed CRS must also be equivalent to the explicitly declared lunar south-polar stereographic CRS:

- lunar spherical radius `1,737,400 m`;
- latitude of origin `-90 deg`;
- true-scale latitude `-90 deg`;
- central meridian `0 deg`;
- meter units.

This prevents an Earth/default CRS or wrong lunar projection from passing merely because a coordinate transform returns finite numbers.

The receipt exports the observed CRS WKT and its SHA-256 so the downstream LL-009L config can pin that exact observation.

## Sampling policy

V1 uses `containing_pixel_center_no_interpolation`:

1. transform the published lunar longitude/latitude into the observed source CRS;
2. locate the containing raster row/column;
3. fail on out-of-bounds or nodata;
4. read elevation and uncertainty from that exact cell;
5. calculate the exact cell-center X/Y;
6. transform the pixel center back to the explicit lunar geographic sphere for audit;
7. fail if the reference-to-center offset exceeds the contract.

No elevation or uncertainty resampling occurs.

## Output

`ll009p.site-anchor-receipt.v1` binds:

- published anchor source and coordinate;
- projected reference X/Y;
- selected row/column;
- pixel-center X/Y and lon/lat;
- reference-to-center offset;
- exact elevation and vertical uncertainty values;
- exact source hashes;
- source-lock and config hashes;
- observed raster metadata and CRS hash;
- Rasterio version;
- a mechanical `ll009l_site_block` containing `x_m`, `y_m`, `elevation_m`, and `vertical_uncertainty_m`.

## Uncertainty semantics

LL-009P preserves the source uncertainty value but does not strengthen its meaning.

For Site01 the total-Z raster is formally RMS error. The emitted site scalar therefore remains RMS evidence. LL-009O is authoritative for whether the resulting horizon may be described as deterministic, risk-qualified, or descriptive only.

## Local logic evidence

The synthetic campaign passed locally with Rasterio 1.5.0 as logic evidence only. It verifies deterministic containing-pixel selection and replay, exact extraction of elevation/uncertainty, the center-offset gate, and fail-closed rejection of a shifted uncertainty transform.

Promoted execution still belongs to the pinned LL-009L GIS evidence lineage rather than the developer's local 1.5.0 environment.

## Non-claims

Passing LL-009P does not select a launch site, establish that R01 is the best point on Connecting Ridge, establish a deterministic terrain uncertainty bound, close unresolved/subpixel terrain, or establish illumination/communications/architecture viability.

It establishes a narrower fact: the first Site01 run can obtain its site state from the same exact NASA raster bytes used by the terrain analysis, rather than from hand-entered coordinates/elevation/uncertainty.
