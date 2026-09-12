# LL-009T — Statistical horizon composition and exact K binding

## Purpose

LL-009T closes the numeric gap between LL-009Q's Site01 clone-ensemble horizon evidence and LL-009S's evidence-aware visibility lane.

The central rule is simple:

> Statistical semantics may be attached to a K-compatible horizon only when the actual emitted horizon numbers were produced from the exact statistical evidence named by the receipt.

The existence of an LL-009Q receipt beside an unrelated LL-009K pack is not sufficient.

## Inputs

T V1 binds exact:

- LL-009T composition policy;
- LL-009Q ensemble policy;
- LL-009Q clone-horizon receipt;
- LL-009O uncertainty-semantics receipt;
- LL-009R spatial-support receipt;
- LL-009M radial-uncertainty receipt;
- the exact LL-009M augmented K input;
- the exact source artifacts referenced by that K input.

Receipt self-hashes and cross-stage byte hashes are checked before composition.

## Ensemble statistic

T V1 supports two explicit modes:

- `empirical_ensemble_quantile` using Q's exact `empirical_cdf_nearest_rank` estimator;
- `finite_ensemble_observed_max` using the largest value actually observed among the exact published clone members.

For quantile mode, the requested quantile must already exist exactly in Q's receipt. T does not interpolate an undeclared quantile.

The checked-in Site01 policy selects `0.99`. With 100 clone members, that remains a finite empirical statistic. It is not silently interpreted as a population-level 99% confidence guarantee.

## Companion terrain

The exact M-augmented K input contains both the Q-covered near layer and the other terrain layers. T removes the Q-covered layer from a temporary canonical copy and invokes the existing LL-009K implementation on the remaining layers.

This is important: T does not implement a second far-field geometry engine.

The companion K horizon therefore remains derived by K's existing maximum-envelope theorem over exact provenance-bound inputs.

T V1 requires every companion layer to have `hard_upper_bound` vertical semantics in LL-009O. Unknown or RMS companion semantics fail closed.

## Observer/site compatibility

LL-009Q uses `same_realization_site_pixel`: each clone changes both surrounding terrain and the observer/site elevation using the same realization.

That creates a non-obvious composition constraint. A far-field K horizon calculated against one nominal site state cannot be combined with Q ensemble statistics as though the observer were statistically identical.

T V1 therefore requires the LL-009O site vertical uncertainty to be `hard_upper_bound` before a Q statistic can be combined with the companion K horizon.

This deliberately means the current Site01 RMS-derived site state is blocked.

A follow-on lane must propagate the same realization-specific observer/site elevation into companion far-field geometry before the real Site01 statistical full-horizon pack can pass.

## Spatial support

LL-009R remains independent from vertical/statistical uncertainty.

For each layer, T preserves the exact R support class. If the R receipt contains an executed `observed_positive_excursion_margin_deg`, T adds exactly that margin to that layer's obstruction before the layer competes for the final maximum.

No margin is invented for `sample_points_only` or `resolution_qualified` evidence.

An empirical multiresolution margin remains empirical; it is not promoted into continuous-hard terrain support.

## Bin composition

For each azimuth bin:

```text
Q obstruction
    = selected exact Q statistic
    + exact R margin, if present

companion obstruction
    = exact re-materialized K companion horizon
    + winning companion layer's exact R margin, if present

final obstruction
    = max(Q obstruction, companion obstruction)
```

Winning provenance is retained for every bin.

## Statistical K binding

A passing output remains schema-compatible with `ll009k.horizon-pack.v1` and adds:

- `statistical_horizon_binding.status = bound`;
- estimator/mode/quantile;
- exact Q policy and receipt hashes;
- exact O/R/M receipt hashes;
- exact M-augmented K input hash;
- exact derived companion-K hash;
- exact T composition-policy hash;
- covered layer IDs;
- observer compatibility rule;
- spatial-support-margin policy;
- exact bin composition rule.

This is the object LL-009S can use to distinguish a genuinely statistical numerical horizon from a sidecar statistical receipt that was never used to produce the horizon numbers.

## Synthetic executed logic evidence

The LL-009T synthetic campaign exercises:

1. a Q empirical quantile winning some azimuth bins;
2. a hard-bound far-field K obstruction winning another bin;
3. exact LL-009R empirical positive-excursion margin application;
4. deterministic replay;
5. rejection of an undeclared empirical quantile;
6. rejection of RMS site uncertainty for companion geometry.

The composition logic was also locally syntax-checked. This is logic evidence, not real NASA Site01 execution evidence.

## Current Site01 state

The checked-in Site01 T policy is **expected to block** today because:

- Site01 site vertical uncertainty inherits RMS semantics from the total-Z product;
- Product 90 far-field `ADJ_ERR` semantics are not yet strong enough to satisfy the V1 hard-bound companion requirement;
- LL-009R still reports non-continuous spatial support.

This is preferable to producing a statistically impressive but internally inconsistent horizon.

## Non-claims

A passing T pack would still not establish:

- a deterministic physical terrain upper bound;
- a population confidence guarantee from a finite clone quantile;
- continuous terrain closure unless R separately establishes it;
- solar power availability;
- RF link quality;
- site suitability;
- architecture superiority;
- construction, launch, or operations authority.
