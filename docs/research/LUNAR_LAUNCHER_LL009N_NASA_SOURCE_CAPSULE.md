# LL-009N — Exact NASA source-byte acquisition capsule

## Purpose

LL-009N is the network/acquisition boundary in front of the existing lunar terrain evidence chain:

`NASA PGDA source -> LL-009N exact bytes -> LL-009L raster materialization -> LL-009M terrain-radial uncertainty -> LL-009K conservative skyline`

It exists so a filename, product page, mutable URL, HTTP header, developer download directory, or rounded website file size can never silently become scientific evidence.

LL-009N establishes **which exact bytes were acquired and can be replayed offline**. It deliberately does not decide whether those bytes are scientifically sufficient for a complete terrain horizon.

## First real-data source plan

`configs/lunar_transport/ll009n_site01_nasa_sources_v1.json` defines two source groups.

### Site01 / Connecting ridge near field

NASA PGDA product 78 publishes 5 m/pixel site GeoTIFFs in south-polar stereographic X/Y coordinates, MOON_ME / DE421. The first horizon-specific pair is:

- `Site01_final_adj_5mpp_surf.tif` — surface elevation;
- `Site01_final_adj_5mpp_toterr.tif` — total-Z RMS elevation uncertainty.

PGDA states that roughly 90% of the 5 m polar LDEM pixels are interpolated. Therefore LL-009N records the nominal pixel spacing only as source metadata; it does not promote “5 m/pixel” into a claim of independent 5 m terrain knowledge.

### Large-area far field

NASA PGDA product 90 publishes cloud-optimized GeoTIFFs in the same south-polar stereographic MOON_ME / DE421 lineage. The first bounded-size far-field set uses the 80 m products:

- `LDEM_80S_80MPP_ADJ.TIF` — surface elevation;
- `LDEM_80S_80MPP_ADJ_ERR.TIF` — surface-height error;
- `LDEM_80S_80MPP_ADJ_EFFRES.TIF` — effective resolution.

This set is intentionally smaller than the 20 m triplet and is useful for exercising the real L/M/K chain without requiring roughly ten gigabytes for the far-field trio alone. It is **not** declared sufficient for a theorem-level complete horizon merely because it is easier to acquire.

The 20 m and 40 m products remain valid future evidence layers. Resolution policy belongs in LL-009L/K study contracts, not in the downloader.

## Plan versus lock

The checked-in JSON is a **source plan**, not a source lock.

Unknown source hashes are represented as `null`. LL-009N never invents a digest from a filename, source-page provenance, an HTTP ETag, or a rounded file size.

On successful acquisition the tool emits two immutable artifacts:

1. `ll009n.nasa-source-lock.v1` — deterministic cryptographic identity: source URL, safe artifact path, exact SHA-256, exact byte size, role, dataset lineage and LL-009L-ready `{path, sha256}` bindings;
2. `ll009n.nasa-transport-receipt.v1` — HTTP transport observations such as final URL, redirect chain, Content-Length, Content-Type, ETag and Last-Modified.

Only the first object establishes byte identity. Transport metadata is useful lineage, but is explicitly non-cryptographic.

## Network restrictions

The V1 acquisition path is stdlib-only and requires:

- HTTPS;
- an explicit host allowlist;
- no URL credentials;
- normal HTTPS port semantics;
- every redirect target to remain HTTPS and allowlisted;
- `Content-Encoding` absent or `identity`, so the hashed stream is the exact transferred representation rather than a transparently transformed payload;
- safe relative output paths with no traversal;
- no-clobber installation;
- exact Content-Length agreement when the server supplies that header.

Changing the allowlist is an explicit source-plan change.

## Atomicity and interrupted runs

Each source is streamed into a temporary file in the destination filesystem, hashed while streaming, flushed and fsynced, then installed with a no-clobber hard-link operation.

If a later file in the same requested plan fails, files installed by that invocation are removed and no source lock is emitted. This prevents an ordinary handled failure from leaving a half-complete acquisition that could be mistaken for a locked source set.

A process kill or machine failure can still leave completed files with no lock. Such files are intentionally not adopted automatically on retry; the operator must remove them or independently establish a new acquisition lineage. LL-009N will not infer provenance from orphaned bytes.

## Offline replay

`verify` performs no network operation. It reads the immutable lock, re-hashes every local file, checks the exact byte count, and emits `ll009n.nasa-offline-verification.v1`.

Replay fails closed on missing files, changed bytes, changed byte counts, duplicate or unsafe lock paths, and malformed digest metadata.

The replay receipt is deterministic for the same lock and artifact root.

## Promotion sets

The source plan exposes two explicit completeness sets:

- `site01-near-field-source-bytes` means only that the exact Site01 elevation + total-Z-RMS pair exists;
- `site01-first-raster-chain-source-bytes` means that the Site01 pair plus the 80 m large-area elevation/error/effective-resolution triplet exists.

Neither set means “complete lunar horizon.” They are source-byte completeness statements only.

## LL-009L handoff

Every lock entry contains:

```json
"ll009l_binding": {
  "path": "nasa/pgda/.../source.tif",
  "sha256": "<exact acquired digest>"
}
```

Those values map directly to LL-009L's required `*_source.path` and `*_source.sha256` fields. LL-009L still independently re-hashes every raster before opening it and still validates CRS, affine alignment, pixel scale, nodata and effective-resolution policy.

LL-009N does not weaken or bypass the pinned LL-009L toolchain. Promoted raster materialization remains CPython 3.13 + Rasterio 1.4.4 with the separately pinned wheel digest and no network access during materialization.

## Continuous-terrain / subpixel closure

LL-009M fixed a real geometric issue: terrain-height uncertainty must follow the terrain point's lunar radial rather than the launch site's local up direction.

A different question remains: does the admitted raster evidence conservatively bound terrain **between** the pixel-center samples used to build the skyline?

The large-area PGDA products provide both surface-height error and effective resolution, which are valuable evidence. But LL-009N does not assume that those two fields automatically prove that no unresolved subpixel ridge or peak can exceed the derived skyline. That needs either:

- a source-supported theorem connecting DEM error/effective-resolution semantics to a continuous terrain upper envelope;
- a conservative horizontal/slope/roughness margin;
- a higher-resolution nested layer that closes the relevant range; or
- an explicit residual-uncertainty/non-claim.

Until that closure exists, the correct downstream phrase is **NASA-terrain-backed raster skyline evidence**, not “the physical horizon is proven complete.”

## Local executed logic evidence

The LL-009N self-test has been executed locally under CPython 3.13. It verifies deterministic offline replay, changed-source rejection, path-traversal rejection, and non-allowlisted-host rejection.

The self-test does not contact NASA and therefore does not establish any real NASA source hash.

## Example commands

Acquisition, on a machine with sufficient storage and network access:

```bash
python scripts/acquire_ll009n_nasa_rasters.py acquire \
  --plan configs/lunar_transport/ll009n_site01_nasa_sources_v1.json \
  --artifact-root evidence/ll009n/site01 \
  --lock-output evidence/ll009n/site01/source-lock.json \
  --transport-output evidence/ll009n/site01/transport-receipt.json
```

Offline replay:

```bash
python scripts/acquire_ll009n_nasa_rasters.py verify \
  --lock evidence/ll009n/site01/source-lock.json \
  --artifact-root evidence/ll009n/site01 \
  --output evidence/ll009n/site01/offline-verification.json
```

Logic campaign:

```bash
python scripts/acquire_ll009n_nasa_rasters.py self-test
```

## Non-claims

Passing LL-009N does not establish site selection, horizon completeness, illumination, Earth/relay visibility, thermal closure, geotechnical suitability, launch/elevator feasibility, economics, construction authority, or safety.

It establishes a smaller but necessary fact: downstream analysis is operating on a cryptographically identified, offline-replayable set of source bytes rather than on an informal filename or mutable download.
