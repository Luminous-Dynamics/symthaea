# LL-009A2 — lunar terrain/site frame bridge evidence

Status: **independent Phase-0 oracle; no real DE421↔DE440 numeric parity claimed yet.**

NASA Goddard PGDA's 5 m South-Pole LOLA terrain products are published in `MOON_ME` tied to DE421. The high-fidelity LL-004E release-state work explicitly uses `MOON_ME_DE440_ME421`. NAIF documents the latter as closely aligned with `MOON_ME_DE421`, but the identifiers represent distinct ephemeris-specific frame realizations and must not be silently relabeled.

## Bridge contract

Given source and destination body-fixed-to-common-inertial rotations at the same epoch/time scale:

```text
r_inertial = R_source · r_source
r_dest     = R_destᵀ · r_inertial
```

The independent oracle in `scripts/ll009a2_lunar_frame_bridge_oracle.py` verifies:

- source/destination orientation snapshots use the same common inertial frame;
- time scales and epochs close exactly within a tiny declared tolerance;
- both rotation matrices are proper orthonormal rotations;
- rigid transforms preserve physical radius/norm;
- the inverse bridge recovers the source vector in synthetic fixtures;
- malformed rotations and epoch mismatches fail closed.

## Native-raster policy

This bridge is deliberately **not** a raster reprojector.

LOLA terrain, slope, and uncertainty rasters should remain in their native DE421 south-polar stereographic representation unless there is a demonstrated reason to produce a separate reprojected derivative. This avoids introducing interpolation/resampling error merely for conceptual tidiness.

At the dynamics boundary, a queried physical terrain/site point can be transformed through explicit 3-D frame reconciliation. For ballistic terrain-clearance work, the preferred early implementation is to transform trajectory query points into the native terrain frame before raster lookup. Any later reprojected raster must be a separately versioned derived artifact with hashes and an explicit error budget.

## Real-data gate

The real DE421↔DE440 transform is intentionally deferred until #1602/#1612 can generate provenance-bound orientation snapshots for both frame lineages. The trade study should then measure:

- angular frame difference;
- lunar-surface displacement magnitude;
- variation with epoch over representative launch windows;
- materiality relative to 5 m raster resolution, LOLA geolocation/height uncertainty, launcher pointing uncertainty, and required terrain-clearance margins.

Only after that measurement may a study use an approximation such as “frame difference negligible for this specific metric and tolerance.”

## Non-claims

This oracle does not reproject a GeoTIFF, select a site, define a launch window, establish terrain accuracy, navigation accuracy, safe corridor clearance, or physical launch authority.
