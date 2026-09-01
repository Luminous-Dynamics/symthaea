# St. Lucia STAC Discovery Execution v1

Status: **pre-run / pre-product-selection**

This document binds the executable catalogue-discovery mechanism to `ST_LUCIA_SENTINEL_PILOT_V1.md`. It does not change the frozen site, dates, cloud threshold, pairing window, baselines, success criteria, or scientific stage gates.

## Executable

Canonical runner:

```text
scripts/research/st_lucia_stac_discovery.py
```

Frozen runner schema/version for the first execution:

```text
schema       symthaea-st-lucia-stac-discovery/v1
tool_version 1.1.0
```

The runner uses only the Python standard library and performs **catalogue discovery only**. It must not download raster assets, thumbnails, previews, quicklooks, or labels.

The runner follows current CDSE STAC conventions used by the frozen protocol:

- Sentinel-2 L2A item assets expose band keys such as `B03_10m`, `B04_10m`, `B08_10m`, `B11_20m`, `B12_20m`;
- Sentinel-1 GRD item properties use `sar:instrument_mode` and `sar:polarizations`;
- orbit metadata such as `sat:orbit_state` is retained in the raw page even though orbit direction is not a v1 selection criterion.

## Qualification before live discovery

Before the first live catalogue request, record:

```text
git rev-parse HEAD
sha256sum docs/research/ST_LUCIA_SENTINEL_PILOT_V1.md
sha256sum scripts/research/st_lucia_stac_discovery.py
python3 -VV
python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)'
```

Run the offline discovery tests:

```text
python3 -m unittest scripts/research/test_st_lucia_stac_discovery.py
```

The offline suite covers deterministic S2 selection, half-open time bounds, required-band metadata, S1 time-distance/earlier/ID tie-breaking, IW + VV/VH requirements, exhaustive pagination, raw-page retention, identical-item deduplication, conflicting duplicate IDs, pagination cycles, and off-origin pagination rejection.

A failing offline test blocks live product selection.

## Live command

Execute from the repository root on the exact qualified commit:

```text
python3 scripts/research/st_lucia_stac_discovery.py \
  --protocol docs/research/ST_LUCIA_SENTINEL_PILOT_V1.md \
  --out /tmp/st-lucia-sentinel-discovery-v1
```

The output directory is intentionally outside the source tree during execution. Raw evidence is reviewed before deciding which evidence artifacts belong in the repository/release capsule.

## Required evidence

The run must retain:

```text
discovery_receipt.json
s2/page-0001.json
s2/page-0001.sha256
...
s1/page-0001.json
s1/page-0001.sha256
...
```

Each page record binds:

- exact GET/POST method;
- exact URL;
- exact POST body when pagination requires one;
- request-body SHA-256;
- raw response byte length;
- raw response SHA-256;
- retrieval timestamp;
- response headers exposed to the client;
- feature count.

The catalogue-snapshot digest deliberately excludes retrieval timestamps and response headers. It commits the ordered raw page digests, byte lengths, and exact requests so identical catalogue bytes under identical requests have the same snapshot identity even when retrieved at a different wall-clock time.

## Pagination and origin policy

The first query and every server-supplied `next` link are restricted to:

```text
scheme https
host   stac.dataspace.copernicus.eu
path   /v1/...
```

Off-origin pagination is a hard failure. Pagination request cycles are a hard failure. Pagination continues until no `rel=next` link remains; a first page or arbitrary `limit` is not an exhaustive candidate universe.

If the same STAC item ID appears more than once with byte-distinct canonical metadata, discovery fails with:

```text
duplicate-id-metadata-conflict
```

Identical duplicate items may be deduplicated only after the complete raw pages have been retained.

## Selection receipt

The machine-readable receipt must expose:

- protocol SHA-256;
- runner schema/version;
- exact discovery AOI;
- exact S2 and S1 query URLs;
- per-modality catalogue-snapshot SHA-256;
- every candidate with eligibility/ineligibility reasons;
- every eligible candidate ID in exact deterministic selection order;
- selected S2 item ID or explicit null;
- selected S1 item ID or explicit null;
- receipt SHA-256.

The complete raw item metadata remains in the retained STAC pages. This includes orbit/platform/processing metadata exposed by the catalogue even when those fields are not product-selection criteria.

The receipt is not a claim that raster bytes were downloaded or that the selected catalogue record is scientifically valid ground truth.

## Null and failure semantics

Use these distinctions:

```text
selected
no-eligible-item-visible
not-run-no-selected-s2
catalogue-query-failed
duplicate-id-metadata-conflict
pagination request cycle
```

`no-eligible-item-visible` means only that no eligible item was visible in the frozen, exhaustively retrieved catalogue snapshot. It is not evidence that the satellite made no physical acquisition.

## Review boundary before asset download

After discovery, stop.

Do **not** download selected product assets until reviewers have checked:

1. protocol hash matches the preregistered document;
2. runner hash/version matches this execution contract;
3. offline tests passed on the exact commit;
4. all pagination pages are present and hashed;
5. candidate ordering reproduces from the retained pages;
6. selection follows the frozen rule exactly;
7. no preview/quicklook inspection influenced selection;
8. null/failure states, if any, are retained rather than relaxed post hoc.

Only then may a separate evidence step acquire exact source assets and bind their byte digests into PP-05 / #202 lineage.
