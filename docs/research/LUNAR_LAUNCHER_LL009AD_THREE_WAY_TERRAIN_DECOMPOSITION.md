# LL-009AD — three-way scale-matched terrain discrepancy decomposition

## Purpose

LL-009AD explains where an LL-009AC `SDEM80 - Product90_80` discrepancy comes from before anyone interprets it as evidence about Product90 uncertainty calibration.

For every exact AC-admitted 80 m support cell, AD enforces the identity

`SDEM80 - Product90_80 = (SDEM80 - Site01_LDEM80) + (Site01_LDEM80 - Product90_80)`.

The two right-hand terms are intentionally described as:

- **SfS increment** — the scale-matched Shape-from-Shading difference relative to the exact Site01 5 m LOLA LDEM;
- **LOLA baseline difference** — the difference between the exact Site01 5 m LOLA LDEM aggregated to 80 m and the separate Product90 80 m LOLA product.

NASA PGDA documents both the Site01 high-resolution LDEM and Product90 large-area DEMs in south-polar stereographic coordinates in the MOON_ME / DE421 frame, while Product104 states that its SDEM retains consistency with the LOLA geodetic reference. AD still requires exact raster-lattice checks; the shared source frame is not used as permission to resample.

Sources:

- Product78 / Site01: https://pgda.gsfc.nasa.gov/products/78
- Product90: https://pgda.gsfc.nasa.gov/products/90
- Product104: https://pgda.gsfc.nasa.gov/products/104

## Exact AC population inheritance

AD does not define a new coarse-cell population.

It reconstructs AC's exact SDEM/Product90 support records and requires both:

- `admitted_cell_count` equality;
- exact equality of the reconstructed AC `cell_record_digest_sha256`.

If even one coarse cell, fine support tile, source value or numerical aggregation differs from the exact AC receipt, AD fails before emitting a decomposition.

## Site01 5 m LDEM co-registration

The Site01 5 m LDEM must have the same fine-lattice matrix as the SDEM. Different extents are permitted, but its origin must be an exact integer fine-pixel offset.

Every AC-admitted SDEM fine tile must be covered by valid Site01 LDEM samples. A subpixel shift or missing Site01 value on an AC-admitted tile fails closed. V1 does not interpolate or reproject.

## Two independent algebraic decompositions

AD repeats the decomposition for both AC aggregation conventions:

1. projected equal-weight fine-cell mean;
2. spherical-stereographic Jacobian area-weighted mean.

For each admitted cell it emits:

- total SDEM-to-Product90 residual;
- SfS increment;
- LOLA baseline difference;
- closure error;
- all three terms normalized by exact Product90 `ADJ_ERR` RMS.

The configured closure tolerance is numerical only; it is not an uncertainty budget.

## Descriptive ancestry diagnostics

AD reports finite-population moments for every component, plus descriptive covariance/correlation between the SfS and LOLA-baseline terms.

For the area-weighted decomposition each cell is also classified as:

- `sfs_dominant`;
- `lola_baseline_dominant`;
- `mixed_equal`.

The classification uses the explicit configured absolute-magnitude tie tolerance. AD also reports whether the two signed components reinforce or cancel one another and preserves the largest total normalized discrepancies with their components attached.

These are **ancestry-aware descriptive diagnostics**, not causal attribution. The three terrain products share LOLA data and processing lineage; component covariance cannot be interpreted as independence or causal effect.

## Why this changes downstream interpretation

Suppose AC finds `|SDEM80 - Product90_80| / ADJ_ERR` substantially larger than one.

Without AD, that result is ambiguous.

- If the SfS increment dominates, the discrepancy is associated mainly with terrain information added by Shape-from-Shading relative to the Site01 LOLA LDEM. That points toward short-scale spatial-support / reconstruction limitations as an important mechanism.
- If the LOLA baseline term dominates, the discrepancy already exists between two LOLA-derived products after scale matching. That points toward cross-product processing/harmonization and uncertainty-model questions rather than simply 'SfS discovered missing terrain.'
- If the components oppose one another, the total residual may hide substantial internal disagreement through cancellation.

AD makes those cases machine-visible without promoting any of them to a causal or probabilistic theorem.

## Local executed logic evidence

The exact production implementation has Git blob:

`1fc61381a72140a9366cf30ee75f46bd90942bc3`.

That byte-identical artifact compiled and passed a synthetic Rasterio 1.5.0 / NumPy 2.3.5 campaign. The campaign verifies:

- deterministic replay;
- zero algebraic closure error in every admitted synthetic cell;
- exact reconstruction of AC's admitted count and cell-record digest;
- deterministic dominance/reinforcement accounting;
- a 1 m subpixel shift of the Site01 5 m LDEM is rejected even though AC itself is unaffected by the near-layer shift.

No real AD receipt is claimed yet.

## Real Connecting Ridge path

1. Produce the real AA/AB SDEM role receipt.
2. Produce the real AC scale-matched Product90 RMS model-check receipt.
3. Bind the exact Site01 5 m LDEM already present in LL-009L.
4. Run AD on the exact AC population.
5. Inspect whether large AC residuals are SfS-dominant, LOLA-baseline-dominant, reinforcing or cancelling.
6. Keep any later uncertainty-calibration or spatial-support promotion in separate receipts.

## Non-claims

LL-009AD does not establish independent ground truth, causal error attribution, an uncertainty multiplier, independence between components, calibrated Product90 probability, a hard terrain bound, continuous-terrain completeness, visibility qualification, mission safety or architecture authority.
