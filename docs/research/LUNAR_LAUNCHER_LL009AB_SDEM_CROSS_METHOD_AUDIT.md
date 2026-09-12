# LL-009AB — SDEM role binding and cross-method terrain audit

## Purpose

LL-009AB is the scientific boundary after LL-009AA's exact-byte archive acquisition. AA intentionally refuses to infer scientific meaning from archive filenames. AB requires a reviewed role map that binds an exact AA manifest member by **path + SHA-256 + documentary basis**, validates its raster contract, and only then allows a cross-method comparison with the existing LOLA/LDEM evidence chain.

The intended real source is NASA PGDA Product 104 / Zenodo record 17954508 for the Connecting Ridge 5 m/pixel Shape-from-Shading products. Product 104 states that the SDEMs use LROC NAC imagery to add short-scale terrain detail where LOLA-based LDEMs are limited by sparse laser sampling and smooth interpolated gaps. The archive includes SDEM elevation, SDEM-minus-LDEM differences, image/illumination support products and other ancillary rasters.

## Phase A — exact scientific role binding

`bind` consumes:

- a passing `ll009aa.sdem-archive-member-manifest.v1`;
- a reviewed `ll009ab.sdem-role-map.v1`;
- the exact AA extracted-root directory.

Supported roles are:

- `sdem_elevation` (required);
- `ldem_reference`;
- `sdem_minus_ldem`;
- `image_coverage`;
- `solar_bin_count`;
- `best_input_resolution`.

Every role-map entry must contain the exact AA member path and SHA-256, a human-reviewable documentary basis, expected pixel scale, exact CRS-WKT SHA-256, and an explicit declaration of whether the member is supposed to be pixelwise comparable with the SDEM.

AB does **not** select roles with a glob, regex, filename heuristic or best-effort search. Members declared pixelwise comparable must match the SDEM's CRS, affine transform, dimensions and pixel registration exactly.

The result is `ll009ab.sdem-role-binding-receipt.v1`.

## Phase B — exact-overlap cross-method audit

`audit` consumes the role receipt and the existing exact LL-009L config. V1 deliberately requires:

`grid_alignment_policy = exact_same_pixel_lattice_no_resampling`

The SDEM may cover only a subset of the LDEM. AB therefore does **not** require equal raster extents. Instead it proves that both rasters occupy the same pixel lattice and that the SDEM origin is an exact integer-pixel offset in the LDEM grid. Only their exact aligned overlap is admitted.

A subpixel offset fails closed. No reprojection, interpolation or hidden resampling is performed.

For the exact overlap AB computes:

1. signed and absolute `SDEM - LDEM` elevation residual statistics;
2. exact reconstruction error against the published `sdem_minus_ldem` raster when that role is bound;
3. normalized residual diagnostics `(SDEM - LDEM) / LDEM_RMS` when the LL-009L layer binds an uncertainty raster;
4. explicitly configured support strata from image coverage, solar-bin count or best-input-resolution rasters;
5. SDEM and LDEM horizons from the **same admitted overlap support points** and the exact Site01 observer state;
6. positive and negative SDEM-minus-LDEM skyline excursions per azimuth bin.

The receipt semantics are:

`empirical_cross_method_terrain_discrepancy`

A positive skyline excursion is direct empirical evidence that the imaging-derived terrain model contains a larger obstruction than the corresponding LOLA surface over the exact admitted support. It is not promoted to a continuous physical-terrain hard bound.

## Calibration boundary

Bertone et al. (2026) report that normalized SDEM-LDEM elevation residuals across their regions are broader than a unit Gaussian, approximately `sigma = 1.4`, suggesting the tested LDEM elevation uncertainties may underestimate those residuals by roughly 40%.

AB intentionally does **not** turn that result, or a local Connecting Ridge ratio, into an automatic uncertainty multiplier. The same paper states that the SDEM was regularized toward the LDEM through the initial-DEM constraint, so SDEM-LDEM residuals are not independent ground truth.

Accordingly, AB labels normalized residuals:

`descriptive_model_check_only_not_a_calibrated_probability_theorem`

AB must not silently rescale Product 90 `ADJ_ERR`, replace LL-009V/W/X semantics, establish a true second-moment upper bound, or manufacture a Gaussian confidence guarantee.

## Local executed logic evidence

The exact committed implementation is Git blob:

`bf199f7f2e94c3d3e4323b98360f1fd41163375d`

That byte-identical artifact passed local Python compilation and the Rasterio 1.5.0 synthetic campaign. The campaign verifies:

- deterministic replay;
- exact AA manifest → reviewed role-map binding;
- exact CRS/pixel-scale validation;
- different raster extents with exact integer-pixel registration;
- a synthetic SDEM-only ridge survives as a positive skyline excursion;
- normalized residual diagnostics bind the exact LDEM RMS pixels;
- wrong role hashes fail;
- a subpixel SDEM origin shift fails instead of being resampled.

This remains **logic evidence**. No real Product 104 archive member has yet been promoted to an AB role because the real 2.2 GB AA archive has not been acquired and manifested in this evidence lineage.

## Real Connecting Ridge promotion sequence

1. Run LL-009AA on `A3CLR22_6_Connecting_Ridge.zip` and obtain its real archive SHA-256 and complete member manifest.
2. Inspect the exact member manifest and Product 104 documentation; create a reviewed role map with exact path/hash/WKT bindings.
3. Run `bind` in the pinned GIS evidence environment.
4. Bind the exact LL-009L Site01 near-field LDEM and RMS sources.
5. Run `audit` with no resampling.
6. Inspect positive SDEM skyline excursions and support strata.
7. Feed only the appropriate empirical discrepancy evidence into an R successor; keep calibration adequacy as a separate theorem/receipt.

## Non-claims

LL-009AB does not establish independent ground truth, continuous terrain completeness, a calibrated joint probability model, visibility qualification, landing-site safety, RF performance, solar power delivery, mission authority or architecture viability.
