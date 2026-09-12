# LL-009M — Terrain-radial uncertainty closure for LL-009L → LL-009K

## Purpose

LL-009M closes a geometric approximation in the LL-009L/LL-009K horizon evidence path without rewriting either open parent PR.

LL-009K's conservative skyline model raises terrain uncertainty along the candidate site's local-up vector. That is convenient and internally consistent with LL-009L's current preselection, but a terrain pixel's elevation uncertainty physically acts along that terrain point's own lunar radial/local-up direction.

For nearby pixels the directional difference is tiny. For distant skyline blockers, Phase-0 promoted evidence should bound it explicitly.

## Independent theorem scan

M reopens the exact raster bytes and LL-009L config and independently scans **every admitted pixel** under the same source/range/effective-resolution/nodata policy.

It does not rely on LL-009L's emitted top-K candidate set to establish the bound.

For a candidate site radial `u_s`, terrain radial `u_t`, and terrain vertical uncertainty `δ`, define:

`d = δ (u_t - u_s)`.

LL-009K's approximate high-terrain point is

`p_a = p + δ u_s`

while the physical terrain-radial high point is

`p_r = p + δ u_t`.

The candidate site is conservatively lowered along its own radial before both LOS vectors are formed.

If `v_a` is the approximate high-terrain/low-site LOS and `D = |d|`, then when `D < |v_a|`, M uses the conservative total LOS angular-separation bound

`b = atan2(D, |v_a| - D)`.

If `D >= |v_a|`, the promoted evidence fails rather than claiming a finite useful angular bound.

Because elevation-angle error cannot exceed total LOS angular separation, M verifies for every admitted pixel that

`e_radial <= e_site_up + b`.

## Uniform layer bound theorem

Let

`B_layer = max_i b_i`

over every admitted source pixel in the layer.

Then for any pixel `i`:

`e_radial(i) <= e_site_up(i) + B_layer`.

LL-009L emits the strongest site-up skyline candidates per azimuth bin. Therefore adding `B_layer` to the layer's existing LL-009K angular margin also bounds a pixel that was omitted by L's top-K JSON reduction:

`e_radial(i) <= e_site_up(i) + B_layer <= max_j e_site_up(j) + B_layer`.

This is the key reason M does not need to rewrite LL-009L's candidate reducer or duplicate every raster pixel into the normalized JSON artifact.

## Lineage checks

M requires:

- exact LL-009L config bytes;
- exact source raster hashes through L's source verifier;
- exact LL-009L normalized pack bytes;
- exact baseline LL-009K wrapper bytes;
- common study/frame/epoch/site lineage;
- the K wrapper to bind the exact L pack SHA-256;
- matching declared layer angular margins.

Any mismatch fails closed.

## Output

M produces two artifacts:

1. `ll009m.radial-uncertainty-receipt.v1`, containing per-layer all-pixel scan statistics, maximum analytic bound, maximum observed exact radial-minus-site-up elevation delta, per-bin bounds, worst-case source row/column and source hashes;
2. an augmented LL-009K input whose layer margin is

`declared_margin + B_layer`.

The original declared margin and the added M radial-uncertainty margin remain recorded separately.

## Executed synthetic evidence

The dependency-free M logic plus Rasterio-backed synthetic campaign has been executed locally against the environment's Rasterio 1.5.0 and passed.

The synthetic fixture deliberately uses a very coarse lunar polar raster and `top_k_per_bin = 1` so the all-pixel theorem scan covers more admitted pixels than LL-009L emits.

The test verifies:

- the radial-direction bound is non-zero for separated terrain radials;
- every exact terrain-radial elevation delta is <= its analytic bound;
- the augmented K angular margin is strictly larger when required;
- admitted source-pixel count exceeds emitted top-K sample count, demonstrating omitted-candidate closure;
- repeat execution is byte-deterministic.

As with LL-009L, this local Rasterio 1.5.0 execution is logic evidence only. Promoted real-data evidence must use the separately pinned raster toolchain lineage.

## Promotion rule

No real PGDA/LOLA horizon artifact should be promoted through LL-009K/J/I until a passing LL-009M receipt exists for every LL-009L raster layer used by the corridor study.

## Non-claims

A passing M receipt establishes only conservative closure of terrain uncertainty direction geometry. It does not establish:

- completeness of the terrain sources;
- source-resolution sufficiency;
- geolocation/frame uncertainty closure beyond separately declared margins;
- site suitability;
- illumination or communications performance;
- corridor safety;
- economic superiority;
- construction or launch authority.
