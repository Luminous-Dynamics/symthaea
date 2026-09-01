# St. Lucia R0 Asset Plan v1

Status: **post-discovery / post-replay / pre-asset-download**

This document freezes the exact asset-key and access-locator selection boundary for the first St. Lucia Sentinel R0 provenance run before any scientific raster asset is downloaded or previewed. It is stacked on the executed discovery/replay lineage in PR #250 and does not alter the already-frozen site, acquisition time, S2 acquisition support set, S1 pairing, baseline ladder, evaluation criteria, or higher-stage claims.

## Frozen evidence inputs

The planner must consume the already-retained local evidence from the executed St. Lucia discovery and replay. It must not query CDSE or any other network endpoint.

Expected evidence identities:

```text
original discovery receipt file SHA-256
bd1c91e4cb92bb6fe51c0b2ec819d7b5a87530b307dbdb4e35631f92f750efe0

original discovery receipt internal SHA-256
16552ad878560bd8df8242c5b4a3966ca3829f4daafea5e6738e8de8b3b60e85

S2 catalogue snapshot SHA-256
41fec8ace5d07c0ab32c71ab352a5dfe2c77b750ee56e96aeed6afc3b1f28f25

S1 catalogue snapshot SHA-256
3f29e3309527ff469ca1fcaa2cfeeec1c2574af2cbe5c6886285a7c8d98cefc7

acquisition-set receipt file SHA-256
50d25f28b8dd2b1c8787b763d0fdb7ddc668a3a6ab3ae50e5ab79ab0934369bc

acquisition-set receipt internal SHA-256
79ce4a14e95e0e7894c8c0684f7e4c6e4344e5bddb804b8381a9476a8de5c29a
```

The frozen S2 acquisition support is:

```text
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVQ_20260701T122756
```

The frozen S1 pair is:

```text
S1C_IW_GRDH_1SDV_20260630T031023_20260630T031048_008329_0107B6_765A_COG
```

## Exact Sentinel-2 asset keys

For **each** selected S2 tile, the R0 acquisition plan must contain exactly these scientific payload assets:

```text
B03_10m
B04_10m
B08_10m
B11_20m
B12_20m
SCL_20m
```

Rationale:

- `B03_10m`, `B04_10m`, and `B08_10m` preserve their native 10 m L2A representation for the initial visible/NIR features;
- `B11_20m` and `B12_20m` preserve the native 20 m SWIR representation rather than silently resampling them during acquisition;
- `SCL_20m` is retained as the explicit L2A scene-classification / validity-mask evidence source. Using it later still requires a separately frozen mask interpretation; merely downloading SCL does not silently define which classes are valid.

For each S2 tile also retain these provenance/processing metadata assets:

```text
safe_manifest
product_metadata
granule_metadata
datastrip_metadata
```

The planner must not include `thumbnail`, TCI preview assets, or the bulk `Product` archive as substitutes for the frozen scientific assets.

## Exact Sentinel-1 asset keys

For the selected S1 GRD item, retain exactly these scientific payload assets:

```text
vv
vh
```

and these calibration/provenance assets:

```text
safe_manifest
schema-calibration-vv
schema-calibration-vh
schema-noise-vv
schema-noise-vh
schema-product-vv
schema-product-vh
```

The planner must not include `thumbnail` or the bulk `Product` archive by default.

Retaining the S1 calibration/noise/product metadata does **not** claim that the first SAR preprocessing implementation is scientifically qualified. It only preserves the source material needed to reproduce and audit later calibration semantics.

## Planner contract

The frozen planner is:

```text
scripts/research/st_lucia_r0_asset_plan.py
schema       symthaea-st-lucia-r0-asset-plan/v1
tool_version 1.2.0
```

### Executed pre-download correction history

The correction history is preserved rather than rewritten:

1. v1.0.0 passed 21/21 then-current offline tests and failed closed on the first real S2 asset with `href is not absolute HTTPS`.
2. v1.1.0 added safe relative-URL resolution and passed 26/26 tests, but the exact same real-data failure remained.
3. A separate zero-network forensic read of the already-frozen STAC pages then established the actual representation: every one of the 29 selected S2/S1 science and metadata assets uses a canonical `s3://eodata/...` locator. No asset bytes were fetched or previewed during any of these steps.
4. v1.2.0 therefore models the actual CDSE EOData access contract directly instead of converting S3 object locators into invented HTTPS object URLs.

The official Copernicus Data Space Ecosystem S3 documentation identifies the default S3-compatible EOData endpoint as:

```text
https://eodata.dataspace.copernicus.eu/
```

and the unified EOData bucket as:

```text
eodata
```

Reference:

- https://documentation.dataspace.copernicus.eu/APIs/S3.html

## S3 locator contract

For an asset whose retained STAC `href` has scheme `s3`, the planner must:

1. preserve the raw `s3://...` locator byte-for-byte in the plan;
2. require scheme exactly `s3`;
3. require bucket exactly lowercase `eodata`;
4. reject S3 userinfo, ports, query strings and fragments;
5. require a non-empty object key;
6. reject repeated slash forms and backslashes rather than canonicalizing them silently;
7. derive and record the immutable tuple:

```text
endpoint = https://eodata.dataspace.copernicus.eu/
bucket   = eodata
key      = exact path after s3://eodata/
```

The endpoint is the access service; the retained `s3://eodata/...` URI remains the scientific source locator. The planner does not claim that the endpoint URL plus object path is a directly retrievable unauthenticated HTTPS URL. Actual acquisition requires a separately reviewed S3 client invocation and CDSE S3 credentials.

For defensive compatibility only, an absolute HTTPS asset href may still be accepted when it is on the pinned STAC host, and a relative href may be resolved only through one unique retained item `rel=self` link on that host. The executed St. Lucia frozen asset surface does not rely on those fallback forms.

## Planner invariants

The planner must:

1. use zero network access;
2. verify the original discovery receipt file hash;
3. verify the acquisition-set receipt file hash;
4. verify the retained STAC page `.sha256` sidecars before parsing them;
5. verify the exact S1/S2 selected IDs from the acquisition-set receipt;
6. locate those exact items inside the retained raw STAC pages;
7. require every frozen asset key above to exist exactly once per selected item;
8. preserve each raw STAC asset href exactly;
9. validate and decompose each approved `s3://eodata/...` locator into the pinned endpoint/bucket/key tuple;
10. preserve MIME type, roles, title and item identity from the frozen STAC bytes;
11. emit all planned entries in deterministic `(collection, item_id, asset_key)` order;
12. content-address the resulting plan;
13. perform **no asset download**.

Any missing required asset, duplicate selected item, changed source receipt, changed raw-page hash, unsafe/unrecognized locator, wrong bucket, malformed S3 key, or selected-ID mismatch is a hard failure.

## Review stop

After generating `asset_acquisition_plan.json`, stop again before downloading anything.

Review must confirm:

- expected evidence hashes;
- exactly two S2 source items and one S1 source item;
- exactly the frozen asset keys above;
- exactly 29 asset entries;
- no thumbnail/preview/TCI scientific substitution;
- no bulk `Product` archive substituted for exact assets;
- exact raw `s3://eodata/...` locators from the frozen catalogue bytes;
- exact S3 endpoint, bucket and object key decomposition;
- deterministic plan hash.

Only after that review may a separate acquisition step fetch bytes. Each downloaded file must be hashed immediately and its byte length, exact source S3 locator, endpoint, client/tool version, response/object metadata where available, and download environment recorded before any raster inspection or feature computation.

Credentials are execution secrets and must never be written into the plan, receipts, logs, command history examples, or repository.

## Claim boundary

This plan establishes **what bytes are permitted to be acquired** for R0. It makes no claim about:

- raster decoding correctness;
- atmospheric/cloud-mask interpretation;
- SAR calibration or terrain correction;
- classification or forecasting skill;
- HDC value;
- semantic compression.
