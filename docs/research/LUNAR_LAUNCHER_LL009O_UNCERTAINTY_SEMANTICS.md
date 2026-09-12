# LL-009O — Terrain uncertainty semantics and claim guard

## Purpose

LL-009O closes a claim-semantics gap that is separate from LL-009M's geometry theorem.

LL-009M answers: **if the admitted terrain uncertainty magnitude is a valid bound, how much angular error is introduced by applying that displacement along launch-site up rather than terrain radial?**

LL-009O asks the logically prior question: **what does the admitted uncertainty magnitude actually mean?**

For Site01, NASA PGDA explicitly labels `Site01_final_adj_5mpp_toterr.tif` as total Z uncertainty, *formally the RMS error*. An RMS error is not a deterministic maximum displacement. Therefore `terrain + 1 × RMS` is not eligible for an unqualified theorem-level `upper_bound` or `conservative physical horizon` claim.

The correct architecture is now:

`LL-009N exact source bytes -> LL-009L raster geometry -> LL-009M displacement-direction closure -> LL-009O uncertainty-semantics gate -> statistical/ensemble closure if required -> LL-009K/J/I`

## Semantic classes

V1 recognizes four explicit classes:

- `hard_upper_bound` — the evidence contract really supplies a deterministic magnitude bound;
- `rms_error` — second-moment/statistical error evidence; a separate risk policy is required;
- `empirical_ensemble` — multiple uncertainty realizations/clones; an explicit empirical statistic and finite-ensemble interpretation are required;
- `unknown` — fail closed for deterministic and risk-qualified horizon claims.

No adapter may silently promote `rms_error`, `empirical_ensemble`, or `unknown` to `hard_upper_bound`.

## Claim classes

The guard distinguishes:

- `deterministic_upper_bound` — permitted only when **every** terrain layer and the site vertical uncertainty are `hard_upper_bound`;
- `risk_qualified` — permitted only after all inputs have at least a known non-unknown uncertainty semantics; LL-009O classification alone does not provide the missing statistical/ensemble calculation;
- `descriptive_only` — may pass while preserving explicit non-claims.

For the checked-in Site01 policy, `deterministic_upper_bound` is deliberately blocked. The near-field total-Z raster is RMS, and the large-area `ADJ_ERR` product remains `unknown` until its exact statistical/hard-bound semantics are source-supported and promoted.

## Cross-stage byte binding

A semantic label is useful only if it applies to the exact bytes that entered the geometry calculation.

`classify_ll009o_uncertainty_semantics.py` therefore requires:

1. the LL-009N cryptographic source lock;
2. the exact LL-009L terrain sample pack;
3. the exact LL-009M radial-uncertainty receipt;
4. a hashed LL-009O semantics policy.

For each layer it requires the uncertainty SHA-256 to be identical across N, L, and M. A correct semantic label attached to the wrong raster fails closed.

The M receipt must also bind the exact L-pack hash.

## Site vertical uncertainty

The same rule applies to the launch-site elevation uncertainty. A scalar in an LL-009L config is not automatically a hard displacement bound.

For the first Site01 policy, the site uncertainty is intended to be sampled from the same total-Z RMS raster and therefore inherits `rms_error` semantics. A future site-anchor materialization receipt should bind the exact source pixel/value rather than asking an operator to type an elevation or uncertainty by hand.

## Why the next statistical closure must rescan source pixels

LL-009L retains only top-K skyline candidates per layer/bin under its selection geometry. That is safe for LL-009M's layer-wide additive theorem because M independently scans every admitted raster pixel and constructs a bound that covers discarded candidates.

A later statistical policy cannot simply multiply the retained top-K uncertainty values by a larger factor. If uncertainty magnitude varies spatially, a pixel that loses under `1 × RMS` can become the worst obstruction under `k × RMS` or an ensemble quantile.

Therefore any confidence-expanded RMS or clone-envelope implementation must either:

- independently rescan every admitted source pixel; or
- prove a layer-wide bound that remains valid for discarded pixels.

## Clone ensemble path

NASA PGDA publishes 100 Site01 clones, and Barker et al. use clone realizations to examine the effect of terrain uncertainty on derived illumination. This is a strong candidate for the first empirical Site01 uncertainty lane.

A future clone implementation should:

- acquire and hash-lock every admitted clone through LL-009N-style receipts;
- preserve clone identity/order deterministically;
- recompute the relevant horizon/visibility observable per clone rather than perturbing only retained top-K cells;
- expose empirical quantiles/maxima separately from deterministic bounds;
- state finite-ensemble tail limitations explicitly.

The clone maximum is the maximum of those 100 realizations, not proof that every physically possible terrain realization is below it.

## Continuous-terrain support remains separate

Even perfect uncertainty-magnitude semantics do not close the spatial-support question: can unresolved terrain between raster support points exceed the pixel-center skyline?

Surface-height error, effective resolution, slope, roughness, nested higher-resolution rasters, and/or an explicit spatial interpolation theorem may contribute to that closure. LL-009O intentionally does not conflate statistical vertical error with subpixel terrain support.

## Site01 reference anchor

NASA's Lunar Surface Data Book identifies R01 / Site01 / Connecting Ridge at latitude `-89.4632 deg`, longitude `-137.49 deg`.

Using the same spherical south-polar stereographic convention already declared by the LL-009L lineage (`R=1,737,400 m`, `lat_0=-90`, `lat_ts=-90`, `lon_0=0`) gives approximately:

- `x = -10999.145 m`
- `y = -11999.255 m`

This is useful as a published reference anchor for the first real Site01 materialization contract. It is **not** yet a full launch-site state: elevation and vertical uncertainty must be sampled from the exact hash-locked raster bytes, with the selected pixel/coordinate transform recorded in evidence.

## Local logic evidence

The LL-009O implementation was exercised locally before publication. The self-test verifies:

1. RMS evidence blocks a requested deterministic-upper-bound claim;
2. the same known RMS semantics can enter the `risk_qualified` classification path without pretending the missing risk calculation has already occurred;
3. true `hard_upper_bound` semantics permit the deterministic class;
4. a mismatch between N/L/M uncertainty hashes fails closed.

This is logic evidence only. It is not real Site01 execution evidence and does not establish the statistical semantics of the large-area `ADJ_ERR` product.

## Non-claims

LL-009O does not choose a Gaussian model, confidence multiplier, one-sided/two-sided convention, per-pixel confidence, familywise whole-sky risk, correlation model, empirical clone statistic, or unresolved-terrain margin.

Its job is narrower and important: make it impossible for those missing choices to disappear behind the word `conservative`.
