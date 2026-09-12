# LL-009R — Unresolved/subpixel terrain-support closure

## Purpose

LL-009R separates a spatial question from the vertical-uncertainty question already handled by LL-009M/O/Q:

> What, if anything, lets a raster-supported skyline make a claim about physical terrain **between** its admitted support points?

An exact NASA byte hash does not answer that question. Neither does a correct RMS/clone model. A raster may be perfectly authenticated while a physically higher feature lies between the support represented by its samples.

LL-009R therefore introduces explicit spatial-support semantics and makes downstream horizon claims conditional on those semantics.

## Current Site01 policy

The checked-in Site01 policy deliberately starts conservatively.

### Near field — Site01 5 m

`site01-near-5m` begins as:

```text
sample_points_only
```

NASA PGDA product 78 states that roughly 90% of the polar 5 m LDEM pixels are interpolated. LL-009Q can propagate the published clone ensemble through every admitted raster pixel, but that still describes uncertainty on the represented raster surface; it is not a theorem that no higher feature exists between the spatial support that generated the raster.

### Far field — 80 m large-area COG

`south-pole-far-80m` begins as:

```text
resolution_qualified
```

Product 90 supplies effective-resolution evidence alongside elevation. LL-009L already uses effective resolution as an admission criterion. R permits that evidence to justify a *resolution-qualified sampled horizon*, but not to become a hard interpolation or Lipschitz theorem by nomenclature alone.

## Support classes

LL-009R uses five explicit classes:

- `continuous_hard_bound` — evidence/theorem establishes a deterministic upper envelope over the continuous terrain represented by the layer;
- `empirical_multiscale_bound` — exact nested-resolution evidence measures the largest positive skyline excursion observed when finer terrain is compared with coarser terrain over a declared overlap;
- `resolution_qualified` — source effective-resolution or equivalent evidence supports admission under a declared resolution policy, without claiming continuous closure;
- `sample_points_only` — the horizon is valid only for the admitted raster support points;
- `unknown` — no promotable spatial-support interpretation is available.

These classes are independent of vertical uncertainty classes such as `rms_error`, `empirical_ensemble`, or `hard_upper_bound`.

## Claim gate

`classify_ll009r_spatial_support.py` classifies a requested downstream horizon claim.

A `continuous_hard_bound` claim passes **only** when every admitted layer is `continuous_hard_bound`.

A `risk_qualified_horizon` blocks `unknown` and `sample_points_only` layers. An empirical multiscale or explicitly resolution-qualified layer may enter such a lane only if its separate statistical/risk policy permits it.

An `empirical_sampled_horizon` can pass when every layer has explicit non-unknown support semantics, but the receipt retains the per-layer support classes so consumers cannot mistake a sampled horizon for a physical upper-envelope theorem.

`descriptive_only` remains available with explicit disclosure.

## Nested-resolution audit

`audit_ll009r_nested_resolution.py` measures a concrete failure mode: a coarser terrain model can miss a skyline peak that appears in a finer exact terrain model.

The audit binds:

- exact SHA-256 for both source rasters;
- their nominal pixel sizes;
- common lunar frame/CRS;
- exact site state;
- explicit lunar reference radius and pole;
- a shared radial overlap;
- a shared azimuth-bin policy.

It independently scans every admitted pixel center in both rasters, computes the horizon in the same site-local physical geometry, and emits for each common azimuth bin:

```text
fine_horizon_deg - coarse_horizon_deg
```

and

```text
positive_excursion_deg = max(0, fine - coarse)
```

The receipt's main statistic is:

```text
max_positive_fine_minus_coarse_deg
```

This is the largest higher-resolution skyline obstruction observed over the exact configured overlap.

## Why the audit is empirical, not a theorem

Even a zero fine-minus-coarse excursion only says that the tested finer raster did not expose a higher pixel-center skyline in the audited overlap.

It does **not** prove:

- that the fine raster itself resolves all physical terrain;
- that no still-finer feature exists;
- that effective resolution is a deterministic support radius;
- that a roughness statistic is a maximum positive excursion;
- that sampled slope is a global/local Lipschitz constant.

Accordingly, a passing nested-resolution audit has semantics:

```text
empirical_multiscale_bound
```

and the classifier may upgrade a coarse layer only as far as that class. It can never use this audit alone to create `continuous_hard_bound`.

## Candidate hard-bound theorem

A real deterministic continuous-terrain closure would need stronger evidence.

For example, suppose every physical point within horizontal support radius `r` of an admitted raster support point is known to satisfy a deterministic gradient bound

```text
|∇z| <= G.
```

Then unresolved positive height relative to that support point is bounded by

```text
Δz <= G r.
```

That height bound can be transformed into a conservative line-of-sight angular margin and maximized over the layer.

The difficult scientific requirement is evidence for `G` over the *continuous support neighborhood*. A slope value sampled at one raster cell, an RMS slope error, or an RMS roughness statistic is not automatically such a bound.

R therefore supports this theorem shape conceptually but refuses to manufacture `G` from weaker products.

## NASA Product 90 auxiliary evidence

The large-area south-pole product family publishes more than elevation and effective resolution. Candidate follow-on evidence includes multi-baseline RMS height-deviation/roughness products, roughness spectra, Hurst exponent products, and related terrain-characterization layers.

These may be valuable for:

- empirical scale-dependence studies;
- selecting where nested-resolution comparisons are most necessary;
- deriving risk-qualified unresolved-support models;
- detecting terrain whose roughness does not converge adequately with scale.

They are not relabeled as deterministic maxima without source-supported methodology.

## Multi-scale convergence campaign

The strongest practical next lane is not one arbitrary coarse/fine pair, but a convergence ladder where exact coverage permits it:

```text
80 m → 40 m → 20 m → 5 m → independent finer validation
```

For every overlapping radial/azimuth region, record positive skyline excursions introduced by each finer level. This yields an empirical function of scale rather than a single comparison.

Useful outputs include:

- fraction of bins changed by finer terrain;
- maximum positive excursion by scale transition;
- distance and azimuth of the newly dominant obstruction;
- whether the identity of the horizon-dominating feature stabilizes;
- relation between observed convergence and PGDA effective-resolution/roughness evidence.

If excursions remain material at the finest available scale, R should keep the physical-completeness claim blocked.

## Synthetic adversarial evidence

The local LL-009R audit self-test deliberately constructs:

- a coarse, flat lunar raster;
- a finer overlapping raster;
- one narrow positive peak that exists only in the fine raster.

The naive coarse pixel-center horizon misses the peak. The R audit detects a strictly positive fine-minus-coarse skyline excursion.

The same campaign verifies:

- deterministic replay;
- exact source-hash checking;
- nodata fail-closed behavior;
- identical lunar CRS requirement;
- finer nominal resolution requirement.

The policy classifier self-test independently verifies that:

- current sampled/resolution-qualified support blocks a continuous hard-bound claim;
- applying a nested audit can upgrade the coarse layer to `empirical_multiscale_bound`;
- that empirical upgrade still cannot satisfy a continuous hard-bound claim;
- a fully synthetic all-hard-bound policy can pass the deterministic gate.

These are logic tests only, not real NASA spatial-support evidence.

## Evidence chain after R

```text
NASA exact terrain bytes
        ↓
LL-009N — source identity
        ↓
LL-009P — exact site state
        ↓
LL-009L — raster materialization / resolution admission
        ↓
LL-009M — radial displacement geometry
        ↓
LL-009O/Q — uncertainty semantics / clone ensemble
        ↓
LL-009R — spatial-support semantics + multiscale audit
        ↓
LL-009K/J — horizon + visibility
        ↓
LL-009I/E/F/G/H — viability, accounting, decisions
```

A downstream receipt should eventually carry both axes explicitly, for example:

```text
vertical_uncertainty_semantics = empirical_ensemble
spatial_support_semantics      = empirical_multiscale_bound
```

rather than collapsing them into one vague word such as “conservative.”

## Non-claims

LL-009R does not claim that the currently available 5 m or 80 m products establish complete continuous lunar terrain.

It does not claim that effective resolution, RMSD roughness, Hurst exponent, sampled slope, or the finite Site01 clone ensemble are deterministic spatial bounds.

It provides the machinery to measure multi-resolution failures honestly, classify the strongest spatial claim supported by the evidence, and fail closed when the physical-horizon theorem is not yet earned.
