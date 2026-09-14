# St. Lucia Sentinel Pilot — acquisition-set amendment v1

Status: **post-discovery / pre-asset-download**

This is a prospective amendment to `ST_LUCIA_SENTINEL_PILOT_V1.md` made **after catalogue discovery but before any Sentinel raster, preview, quicklook, thumbnail, or scientific feature was downloaded or inspected**.

It does not overwrite the original discovery result. The original v1.1 receipt remains evidence of the item-level rule that exposed this modelling gap.

## Trigger

The exact preregistered discovery run at source head:

```text
71ef19e6a35e02b631bfd3ba69b5e781decb70c0
```

returned two eligible Sentinel-2 L2A items at the earliest acquisition instant, representing adjacent tiles:

```text
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVQ_20260701T122756
```

The v1 rule selected the lexicographically first item (`T36JVP`). That rule is deterministic, but it makes the scientific spatial support of an AOI depend on tile naming when one physical acquisition is partitioned into multiple catalogue items.

No imagery was inspected in discovering this issue; it is visible from catalogue structure alone.

## Frozen source evidence

This amendment is bound to the original live discovery lineage:

```text
protocol_sha256
55d16c7d29b03030b2e53ad93b3e679035ce13169c9fdf1c40ac86549dbebd41

original_discovery_receipt_internal_sha256
16552ad878560bd8df8242c5b4a3966ca3829f4daafea5e6738e8de8b3b60e85

original_discovery_receipt_file_sha256
bd1c91e4cb92bb6fe51c0b2ec819d7b5a87530b307dbdb4e35631f92f750efe0

s2_catalogue_snapshot_sha256
41fec8ace5d07c0ab32c71ab352a5dfe2c77b750ee56e96aeed6afc3b1f28f25

s2_raw_page_sha256
c233f9e85450705f7173f5af16eee226189df8263f6af71a8000f7f773a39fd0

s1_catalogue_snapshot_sha256
3f29e3309527ff469ca1fcaa2cfeeec1c2574af2cbe5c6886285a7c8d98cefc7

s1_raw_page_sha256
ad7932be016575c61a58dea736d45a9331a8b93a3aed39ea381bb0dacae04aac
```

Any mismatch blocks replay.

## Amended Sentinel-2 support rule

The frozen **earliest-acquisition rule does not change**.

Selection now has two levels:

1. Apply the original v1 eligibility rules (collection, half-open July 2026 interval, cloud threshold, required band metadata).
2. Require a valid STAC item `bbox` whose bounding box intersects the frozen St. Lucia discovery AOI.
3. Determine the earliest eligible acquisition datetime exactly as v1 did.
4. The selected Sentinel-2 source support is **every eligible catalogue item at that exact earliest acquisition datetime**, ordered lexicographically by exact STAC item ID.

The previous lexicographic item-ID rule is retained only as an ordering rule inside the acquisition support set; it no longer discards same-acquisition tiles.

For the frozen catalogue snapshot, the expected acquisition support set is therefore:

```text
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVP_20260701T122756
S2C_MSIL2A_20260701T073611_N0512_R092_T36JVQ_20260701T122756
```

This is a source-set decision, not a mosaic/resampling decision. Each tile remains an independently content-addressed source product. Any later mosaic, reprojection, resampling, clipping, masking, or band alignment must create explicit derived-artifact lineage.

## Sentinel-1 rule

The Sentinel-1 pairing rule is unchanged because every selected S2 tile above shares the exact same acquisition datetime.

Replay must reproduce the original deterministic S1 selection from the frozen S1 page:

```text
S1C_IW_GRDH_1SDV_20260630T031023_20260630T031048_008329_0107B6_765A_COG
```

No new catalogue query is permitted for this amendment.

## Replay-only execution

Canonical tool:

```text
scripts/research/st_lucia_stac_replay_v1_2.py
```

The replay tool performs **zero network requests**. It consumes the original discovery directory, verifies the original receipt and every retained page against the hashes above, reconstructs candidates from the exact raw pages, applies this amendment, and writes a new deterministic `acquisition_set_receipt.json`.

The original `discovery_receipt.json` and raw pages are never modified.

## Scientific boundary

This amendment changes only spatial source support at the already-selected earliest acquisition instant.

It does not change:

- site/AOI;
- July 2026 interval;
- cloud threshold;
- required optical bands;
- earliest-acquisition criterion;
- Sentinel-1 ±72 h pairing rule;
- baselines;
- Train/Calibration/Evaluation separation;
- metrics;
- promotion rules;
- R0/R1/R2/R3 stage boundaries.

It grants no predictive-skill, classification, forecasting, compression, or subsurface claim.

## Asset-download gate

Raster/preview download remains blocked until:

1. the replay tool's offline tests pass;
2. replay succeeds against the exact original discovery directory;
3. all original receipt/page hashes verify;
4. the amended acquisition-set receipt contains exactly the expected two S2 item IDs above and the unchanged selected S1 item ID;
5. the amended receipt is reviewed and retained alongside the original v1 receipt.
