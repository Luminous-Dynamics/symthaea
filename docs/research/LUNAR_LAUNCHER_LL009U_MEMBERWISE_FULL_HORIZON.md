# LL-009U — Same-realization memberwise full-horizon ensemble

## Purpose

LL-009U closes the observer-correlation gap deliberately left by LL-009T V1.

LL-009Q's `same_realization_site_pixel` policy changes the Site01 observer elevation with the same clone realization that changes near terrain. U therefore recomputes the companion far-field skyline against **each exact Q member's observer state** before any finite-ensemble statistic is taken.

## Ordering theorem

When near and far obstructions both vary by realization, the scientific observable is:

```text
H_member(bin) = max(H_near_member(bin), H_far_member(bin))
H_stat(bin)   = statistic_members(H_member(bin))
```

It is not generally valid to compute marginal statistics first:

```text
statistic(max(A,B)) != max(statistic(A), statistic(B))
```

The U self-test contains an explicit counterexample: at the 0.75 nearest-rank quantile, two marginal sequences each have quantile 0 while the quantile of their memberwise maximum is 10.

## Near-field authority

U does not reconstruct the Site01 clone DEM a second time. The exact passing LL-009Q receipt supplies, in stable member order:

- each member's near-field horizon by azimuth bin;
- the member-specific Site01 elevation;
- exact Q policy/source/L-config lineage.

U verifies Q's self-hash and those exact bindings before use.

## Far-field authority

For every Q member, U independently scans **all admitted far-field raster cells** from the exact LL-009N/L source bindings. It does not reuse LL-009L top-K candidates selected for the nominal observer.

The far terrain point is raised by its admitted hard vertical bound along the terrain's lunar radial direction, while the observer position uses the Q member's exact site elevation. The resulting line of sight is converted to the common local basis and accumulated into the member/bin maximum.

This means observer-height changes are allowed to change which far terrain cell wins.

## Companion uncertainty contract

U V1 permits the companion layer only when LL-009O classifies its vertical uncertainty as `hard_upper_bound` **and** the U policy declares:

```text
companion_uncertainty_direction = terrain_radial
```

The word `error` in a source filename is never interpreted as a hard radial bound.

The current NASA Product 90 `ADJ_ERR` evidence therefore remains blocked: the verified publication material labels it `surface height error`, but does not yet establish the hard-bound semantics U V1 requires.

## Spatial support

LL-009R remains an independent axis. U preserves each exact R support class and applies `observed_positive_excursion_margin_deg` only when that executed margin is present in the R receipt.

Memberwise statistical composition does not upgrade `sample_points_only`, `resolution_qualified`, or `empirical_multiscale_bound` evidence into continuous terrain closure.

## K-compatible output

A passing U result remains `ll009k.horizon-pack.v1` compatible and carries `ll009u.memberwise-statistical-horizon-binding.v1`, including exact hashes for:

- U policy;
- Q policy + Q receipt;
- O uncertainty receipt;
- R spatial-support receipt;
- source lock;
- LL-009L config;
- exact far elevation/uncertainty/effective-resolution bytes.

It also emits the per-member near, far, and full horizons so the ordering can be audited directly.

## Local executed logic evidence

The exact committed script passed locally under Rasterio 1.5.0 and Python compilation. The campaign verifies:

1. the memberwise-max ordering counterexample;
2. deterministic replay;
3. same-realization Site01 observer propagation;
4. all-cell far-field rescanning;
5. complete member/bin horizon coverage;
6. exact lunar CRS/affine/source-hash gates;
7. explicit terrain-radial hard-bound semantics;
8. fail-closed rejection when the far layer is changed to `unknown` uncertainty semantics.

The committed GitHub blob is byte-identical to the locally executed script.

## Current Site01 result

**Expected to block.** U removes the observer/site RMS correlation blocker, but Product 90 far-field vertical-error semantics are still not source-qualified as the required hard radial bound (or as an explicit compatible statistical model).

That is a scientifically useful result: the remaining blocker is now narrow and named instead of hidden inside the horizon calculation.

## Non-claims

A passing U receipt would remain finite-ensemble terrain evidence. It would not by itself establish deterministic terrain completeness, population confidence, RF link quality, delivered solar power, site suitability, architecture superiority, or construction/operations authority.
