# LL-009W — Distribution-free familywise far-field RMS horizon

LL-009W asks a deliberately narrow question: **what whole-far-horizon upper envelope can be justified from Product 90 RMS terrain-height uncertainty alone, without inventing a Gaussian tail or independent-pixel model?**

It consumes an exact passing LL-009V receipt plus the exact LL-009L far-field elevation, RMS and effective-resolution rasters. V supplies the source meaning (`rms_error`); W supplies a separate statistical theorem. Keeping those roles separate prevents a policy choice from being confused with NASA source semantics.

## Theorem

For admitted pixel `i`, let `e_i` be scalar surface-height error and `s_i` its Product 90 RMS scale. W makes the explicit model-conditional statement

`E[e_i^2] <= s_i^2`.

Markov's inequality on `e_i^2` gives

`P(e_i > k s_i) <= P(|e_i| >= k s_i) <= 1/k^2`.

For `N` admitted raster support points, the union bound requires no independence:

`P(any represented admitted point exceeds z_i + k s_i) <= N/k^2`.

Given familywise exceedance budget `alpha`, W derives—never accepts as a manual override—

`k = sqrt(N / alpha)`.

The checked-in Site01 policy uses `alpha = 0.01` as a **diagnostic 99% represented-support envelope**, not as an accepted lunar operations risk threshold.

## Two-pass execution

W deliberately does not calculate the multiplier from LL-009L top-K candidates.

1. Reopen and hash-check exact elevation/RMS/effective-resolution rasters.
2. Reapply LL-009L radial-range, effective-resolution and nodata admission rules to **every source pixel** and freeze the exact admitted count `N`.
3. Derive `k = sqrt(N/alpha)`.
4. Independently rescan the same admitted population.
5. Apply positive scalar height margin `k * RMS` along the local lunar radial.
6. Compute the per-bin upper skyline.
7. Require the second-pass admitted count to equal the theorem-pass count exactly.

This ordering prevents the risk population from being defined after seeing convenient horizon candidates.

## What W proves

Conditional on the RMS second-moment model being valid at every admitted raster support point, the probability that **any represented admitted support point** exceeds the W positive-height envelope is no greater than `alpha`.

No Gaussian distribution and no inter-pixel independence are used.

## What W does not prove

W is not a deterministic terrain bound. It does not cover unresolved physical terrain between raster support points; LL-009R remains authoritative there. It does not close Site01 observer-height uncertainty; the nominal observer in the W receipt is only a geometry reference, while LL-009U propagates Q's member-specific observer state. It is not a site, power, communications, operations or architecture qualification.

Because `k` grows as `sqrt(N)`, a large real far-field raster may produce an enormous envelope. That outcome is scientifically useful: it quantitatively demonstrates how little joint whole-horizon information is available from marginal RMS values alone, and therefore motivates a source-supported covariance, correlated-field or ensemble model rather than an arbitrary `2σ`/`3σ` convention.

## Local logic campaign

The exact algorithm was exercised on synthetic GeoTIFFs under Rasterio 1.5.0. The campaign checks deterministic replay, exact `sqrt(N/alpha)` arithmetic, `sqrt(N)` scaling when the admitted population changes, complete bin coverage, rejection of manual multiplier overrides, source-hash drift, and an attempted Gaussian-policy substitution.

Real NASA promotion still requires exact LL-009N source bytes, LL-009V qualification against those exact bytes, and execution under the pinned GIS evidence environment.
