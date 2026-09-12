# LL-009L — Native LOLA GeoTIFF/COG materialization

## Purpose

LL-009L is the offline evidence adapter between NASA PGDA/LOLA South-Pole raster products and the dependency-free LL-009K conservative horizon reducer.

It exists so GIS-library behavior, source-raster metadata, pixel registration, nodata, and multi-resolution extraction cannot be hidden inside the horizon/economics code.

## Current NASA data contract

The intended real-data inputs are NASA PGDA/LOLA products in native South-Pole stereographic coordinates and MOON_ME / DE421 lineage:

- 5 m/pixel site products: surface elevation, slope, total-Z RMS uncertainty, slope RMS uncertainty, count maps and clones;
- large-area South-Pole cloud-optimized GeoTIFFs at multiple pixel scales, including elevation and effective-resolution products;
- pixel-center semantics where published by the source product.

The source raster remains native evidence. LL-009L does not rewrite it into a new GeoTIFF.

## Runtime boundary

Rasterio/GDAL is **not** a Symthaea runtime dependency.

`materialize_ll009l_lola_cog.py` imports Rasterio lazily only when an offline materialization run is requested. The normalized output is plain canonical JSON and is consumed by LL-009K without any GIS dependency.

The first promotable toolchain lineage is declared in `configs/lunar_transport/ll009l_raster_toolchain.json`.

## Source-byte requirements

Every elevation, vertical-uncertainty and optional effective-resolution raster is referenced by:

- a safe relative path inside an evidence root;
- exact SHA-256 of the materialized file bytes;
- a declared layer/range policy in the LL-009L config.

The V1 materializer itself does not download remote data. Network acquisition is a separate evidence-materialization step. This keeps promoted runs replayable with `--offline` exact source bytes.

## Raster alignment

Elevation, uncertainty and effective-resolution rasters for one layer must have exactly matching:

- width/height;
- CRS;
- affine transform;
- pixel scale;
- single-band semantics.

The config additionally pins the SHA-256 of the observed source CRS WKT.

A shifted uncertainty grid fails rather than being resampled.

V1 performs **no source-cell resampling**.

## Projection contract

The LL-009L config explicitly declares:

- `projection = south_polar_stereographic`;
- lunar reference radius;
- central meridian;
- true scale at the pole;
- pixel registration convention;
- native frame lineage.

The source raster CRS is observed and hash-checked. Pixel-center X/Y coordinates are transformed to an explicit lunar geographic sphere with the declared radius, then converted to Moon-centered 3-D physical vectors.

No Earth ellipsoid/default geographic CRS is permitted by policy.

For a DE440 study, the resulting native DE421 vectors still require the existing explicit DE421→DE440 frame-reconciliation lineage. LL-009L does not relabel them.

## Range-aware COG/block access

For each layer, the config declares a radial band `[min_range_m, max_range_m]` around the study site.

LL-009L iterates raster block windows and reads only blocks whose bounds intersect the outer-range bounding square. Pixels are then filtered by exact projected radial range.

This gives the same code path for ordinary tiled GeoTIFFs and COGs while avoiding whole-raster memory loading.

The source file itself remains an exact locally materialized artifact for promoted evidence; remote HTTP range reads are not treated as equivalent to an exact source-byte hash in V1.

## Multi-resolution evidence

A layer may declare an effective-resolution raster and a maximum permitted effective resolution for its radial band.

Pixels whose effective-resolution evidence is worse than the declared policy are not admitted.

Missing required effective-resolution evidence fails closed.

This supports the intended hierarchy:

- local 5 m site evidence for nearby skyline structure;
- coarser large-area COG evidence for distant blockers where permitted by the study contract.

Resolution does not determine the winning skyline obstruction; geometry does.

## Candidate reduction

A naive 5 m annulus can contain millions of pixels, so LL-009L does not emit every admitted raster cell as JSON.

For each source layer and azimuth bin, it retains the top `K` candidates by the same conservative high-terrain/low-site selection geometry used by LL-009K, preserving:

- source row/column;
- pixel-center X/Y;
- longitude/latitude in the explicit lunar sphere;
- Moon-centered 3-D position;
- surface elevation;
- vertical uncertainty;
- effective resolution;
- projected range;
- selection azimuth/elevation;
- deterministic rank.

K remains authoritative: it re-hashes the normalized L pack and independently recomputes the 3-D conservative horizon across all layers/candidates.

## Nodata

The V1 promoted policy is deliberately conservative: nodata encountered inside a required layer annulus causes failure rather than being interpreted as zero height or interpolated implicitly.

Future source-specific policies may relax this only with an explicit new evidence contract.

## Synthetic executed evidence

The LL-009L self-test has been executed locally against Rasterio 1.5.0 as **logic validation only**. It verifies:

1. a tiny synthetic lunar polar-stereographic GeoTIFF can be read through the same block/window path;
2. a deliberately raised skyline cell survives deterministic top-K candidate selection;
3. exact replay is byte-deterministic;
4. a shifted uncertainty-grid transform fails alignment;
5. nodata inside the required annulus fails closed.

This local 1.5.0 run is not the promotable dependency lineage. Promoted evidence must use the separately pinned Rasterio toolchain receipt.

## Toolchain lineage

The first promotable Linux x86-64 lane pins:

- CPython 3.13;
- Rasterio 1.4.4;
- wheel `rasterio-1.4.4-cp313-cp313-manylinux_2_28_x86_64.whl`;
- SHA-256 `c072450caa96428b1218b030500bb908fd6f09bc013a88969ff81a124b6a112a`.

The pin is intentional: exact evidence provenance is preferred over silently following the newest GIS package. An upgrade creates a new lineage and parity campaign.

## Output

`ll009l.terrain-sample-pack.v1` binds:

- study/frame/epoch/site lineage;
- projection contract;
- exact config hash;
- actual Rasterio version;
- exact source-raster hashes;
- observed raster metadata including CRS WKT and hash;
- scan/admission/nodata statistics;
- selected provenance-rich samples;
- canonical receipt hash;
- explicit non-claims.

The tool can also emit the `ll009k.horizon-materialization-input.v1` wrapper. That wrapper treats the exact LL-009L pack as K's hashed source artifact.

## Non-claims

Passing LL-009L establishes only deterministic raster evidence materialization. It does not establish:

- that a NASA site is selected for infrastructure;
- that a raster is complete enough for a particular range policy;
- site suitability;
- illumination or communications availability;
- thermal closure;
- geotechnical suitability;
- corridor safety;
- launcher/elevator economics;
- release/construction authority.

Those remain downstream gates.
