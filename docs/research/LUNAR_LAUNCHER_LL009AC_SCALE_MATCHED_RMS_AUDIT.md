# LL-009AC — scale-matched SDEM audit of Product 90 RMS uncertainty

## Purpose

LL-009AC tests the **80 m Product 90 RMS model actually consumed by LL-009V/W/X** using the independent Connecting Ridge Shape-from-Shading terrain evidence introduced by LL-009AA/AB.

It deliberately does not transfer the native 5 m SDEM–LDEM residual dispersion directly to Product 90. The audit first places the SDEM and Product 90 on the **same exact 80 m support cell** and only then forms residual/RMS diagnostics.

NASA PGDA Product 104 states that the SDEMs add LROC-NAC-derived short-scale terrain detail while retaining consistency with the LOLA geodetic reference. NASA polar stereographic products use a 1737.4 km reference radius with true scale at the pole. Those source statements motivate the checked-in geodetic contract; the real evidence lane must still bind the exact raster metadata and reviewed LL-009AB role receipt.

Sources:

- NASA PGDA Product 104: https://pgda.gsfc.nasa.gov/products/104
- Bertone et al. 2026, DOI 10.3847/PSJ/ae5b70
- NASA PGDA lunar polar projection documentation: https://pgda.gsfc.nasa.gov/products/69

## Exact nesting theorem

V1 performs **no reprojection and no interpolation**.

Let the fine SDEM affine lattice matrix be `M_f` and the Product 90 lattice matrix be `M_c`. For configured integer scale ratio `N`, AC requires

`M_c = N M_f`

within the strict affine tolerance, and the Product 90 raster origin must map to an integer fine-pixel coordinate under the inverse SDEM affine transform.

For the real Site01 policy:

`N = 80 m / 5 m = 16`.

Every admitted Product 90 cell therefore has an exact `16 × 16 = 256` fine-cell support tile. Boundary cells without the complete tile, fine nodata, coarse nodata and non-positive RMS cells are excluded with explicit receipt counts rather than filled.

If Product 104 and Product 90 do not actually satisfy this lattice theorem, AC V1 **blocks**. A later explicit polygon-overlap/reprojection theorem would be required instead.

## Two averaging conventions

Product 90 uncertainty is described as RMS uncertainty in the gridded value relative to the true average height within the map pixel. The averaging convention is therefore evidence-bearing.

AC reports two deterministic SDEM aggregates for every admitted coarse cell.

### Projected-cell equal-weight mean

The arithmetic mean of all exact fine samples in the complete `N × N` tile.

### Spherical-surface-area-weighted mean

For the canonical spherical stereographic projection with reference radius `R` and unit scale at the pole, the surface-area element corresponding to projected area `dx dy` is

`dA_surface = dx dy / (1 + (x²+y²)/(4R²))²`.

AC integrates this Jacobian over every fine projected cell with deterministic tensor Gauss–Legendre quadrature. The checked-in Site01 policy compares order 3 and order 5 integration and fails if either the aggregate area or the resulting weighted mean exceeds its convergence tolerance.

This is intentionally stronger than estimating a pixel's spherical area from a polygon through its transformed corners: inverse-projected stereographic pixel edges are not generally geodesic arcs.

AC records

`aggregation_definition_sensitivity_m = area_weighted_mean - projected_equal_weight_mean`

so the consequence of the averaging convention is visible rather than hidden.

## Model-check diagnostics

For every admitted Product 90 cell, the deterministic record digest binds:

- Product 90 elevation;
- Product 90 `ADJ_ERR` RMS;
- exact fine-cell count;
- projected equal-weight SDEM mean;
- spherical-area-weighted SDEM mean;
- both residuals and normalized residuals;
- SDEM min/max/range;
- within-cell SDEM RMS relief;
- area-quadrature convergence diagnostics.

The receipt emits population moments for both normalized-residual definitions and fractions with absolute residual greater than 1×, 2× and 3× the nominal Product 90 RMS scale.

It also emits the 20 largest absolute area-weighted normalized residual cells and a canonical digest over every admitted coarse-cell record.

## Spatial dependence boundary

Adjacent Product 90 cells are spatially correlated. LL-009AC therefore does not label the admitted cell count as an IID sample size.

The audit produces deterministic non-overlapping geographic block summaries using the configured coarse-cell block width. These are descriptive locality diagnostics only: no bootstrap interval, effective sample size or independence theorem is inferred.

## Calibration semantics

The receipt class is

`scale_matched_cross_method_rms_model_check`.

That is stronger evidence about the **specific RMS field used by W/X** than a native-5 m comparison, but it remains a model check.

Even if the normalized residual RMS is greater than one, AC does not automatically turn that number into a Product 90 multiplier. The SDEM is not independent ground truth, and the Bertone et al. workflow regularizes the solution toward the input LOLA surface. A separate calibration-adequacy theorem would be needed before W/X could claim a calibrated probability guarantee rather than a model-conditional stress/risk envelope.

Product 90 `ADJ_ERR` therefore remains `rms_error`, never `hard_upper_bound`.

## Local executed logic evidence

The production implementation was compiled and exercised with Rasterio 1.5.0 / NumPy 2.3.5 against a synthetic exact fine/coarse hierarchy.

The campaign verifies:

- deterministic replay;
- exact integer scale matching;
- incomplete boundary support is excluded and counted;
- projected normalized residual arithmetic reproduces the constructed truth exactly;
- area-weighted diagnostics and order-3/order-5 convergence;
- deterministic spatial block summaries;
- fractional coarse-grid shifts are rejected instead of resampled;
- an attempted V semantic promotion from `rms_error` to `hard_upper_bound` is rejected.

The exact tested production file has Git blob:

`db4fb90334b9c901d3148b738ee1e932d03bc6b0`.

No real Product 104/Product 90 AC receipt is claimed yet.

## Real Connecting Ridge path

1. Acquire and hash the real Product 104 Connecting Ridge archive with LL-009AA.
2. Bind the exact SDEM member with LL-009AB.
3. Materialize the exact Product 90 80 m elevation and `ADJ_ERR` sources through the existing N/L/V lineage.
4. Run AC with `ll009ac_site01_scale_matched_rms_v1.json`.
5. If the 5 m and 80 m lattices do not nest exactly, stop; do not resample silently.
6. If they do nest, inspect projected-vs-area aggregation sensitivity, normalized-residual dispersion, spatial blocks and worst cells.
7. Keep any future probability-calibration promotion in a separate receipt/theorem.

## Non-claims

LL-009AC does not establish independent ground truth, a universal uncertainty multiplier, a hard terrain bound, continuous-terrain completeness, Q×far joint probability, visibility/site qualification, mission safety or architecture authority.
