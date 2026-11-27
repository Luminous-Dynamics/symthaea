# MEL-HEAR-004 — Paired source-work-clustered inference v1

**Status:** synthetic-first Rust comparator; exact-head qualification pending  
**Date:** 27 November 2026  
**Owner reducer:** #13947 / `music_performance_identity`  
**Related:** #14012

## 1. Why a separate comparison layer is needed

The property-relative musical identity reducer reports useful per-item, per-property and per-factor outcomes. A model-vs-model study still needs uncertainty estimates that respect the sampling design.

Multiple excerpts, takes, windows and annotations from one composition/source work are correlated. Treating each triad as an independent Bernoulli observation inflates the effective sample size. A bootstrap over individual triads would repeat that mistake.

This module composes the existing reducer twice and performs only the paired statistical comparison. It does not own musical semantics, decode answer letters, verify audio, or run models.

## 2. Estimand: equal-source-work mean

The primary estimand gives every independent source-work cluster equal weight.

For each source work, compute within-cluster means for:

- Performance-STAY false-difference rate;
- Musical-FLIP miss rate;
- joint exact CLEAN + Performance-STAY + Musical-FLIP rate.

Then average those means across source works. A source work with ten excerpts contributes one cluster mean, just as a source work with one eligible excerpt contributes one cluster mean.

This is intentionally not an item-weighted mean. It prevents heavily excerpted compositions from dominating the estimand.

## 3. Paired cluster bootstrap

For each replicate:

1. Draw exactly (K) source-work cluster indices with replacement, where (K) is the observed number of independent clusters.
2. Use the exact same sampled indices for baseline and candidate.
3. Include all scored triads belonging to each sampled cluster through that cluster's precomputed within-work mean.
4. Compute candidate improvement on each metric.

Improvement sign is consistent:

- STAY false-difference: baseline error minus candidate error; positive is better.
- Musical-FLIP miss: baseline error minus candidate error; positive is better.
- Joint exact: candidate rate minus baseline rate; positive is better.

The implementation uses a fixed-seed SplitMix64 sampler and an interpolated percentile interval. The PRNG is for reproducible resampling only, not cryptographic security.

The initial policy requires at least 30 distinct source-work clusters and at least 100 bootstrap replicates. The synthetic harness uses 2,000 replicates. Production studies should preregister a larger replicate count, seed, confidence level and multiplicity policy before looking at candidate outcomes.

## 4. Exact pairing gate

Each baseline and candidate case is independently passed through the #13947 production reducer. Before comparison, the wrapper verifies that both cases share:

- benchmark/schema/version, dataset revision and evaluator-protocol revision;
- exact item and pair identity sets;
- source-work lineage, target property, target-property reference, pairing manifest and declared STAY/FLIP factors;
- each arm's exact audio digest, question bytes and digest, semantic A/B mapping and expected relation;
- take, performance, performer, instrument-family and renderer identities.

Only model/prompt/runtime/decoding identities and model answers may differ. Missing, duplicate or substituted items fail before any bootstrap calculation.

## 5. What the report means

The report contains baseline and candidate cluster-average point estimates plus three separate paired candidate-improvement intervals. No composite score can hide a STAY/FLIP trade-off.

`SourceWorkClusterFloorMet` means only that the declared cluster-count floor is reached. It does **not** mean the study is scientifically confirmatory. The sampling frame, work-lineage assignments, annotation quality, independence assumptions, preregistration, power and multiplicity controls still require independent qualification.

Below 30 clusters the tool emits descriptive equal-work estimates and `InsufficientSourceWorkClusters`, but no bootstrap intervals.

The report unconditionally leaves source lineage authentication, target-property annotation authentication, audio-byte recomputation, model invocation and population generalization false.

## 6. Synthetic controls

The 30-item fixture has 30 unique synthetic source-work clusters and 90 distinct synthetic audio digests. Baseline and candidate have identical trial structure and differ only in their declared run identity and answers. The fixture deliberately gives the candidate different STAY and FLIP error patterns so that the separate directions cannot collapse into one headline score.

Tests cover:

- deterministic replay under a fixed seed;
- exact zero effect when candidate answers equal baseline answers;
- item-order invariance;
- source-work lineage and audio substitution rejection;
- suppression of intervals below 30 clusters;
- equal-source-work weighting when one cluster receives an extra item.

The fixture's 30 synthetic clusters exercise the algorithm and reporting path. They are not 30 real compositions, do not prove independent sampling, and do not support a real-world performance or SOTA claim.

## 7. Next scientific study

Before a real model comparison:

1. Freeze a rights-reviewed source-work sampling frame and independent performer/work splits.
2. Collect musician labels with separate confidence and disagreement records.
3. Pre-register primary target properties, STAY/FLIP factors, exclusion policy, metric directions, model budgets and prompt/runtime identities.
4. Run the same exact items through baseline and candidate models.
5. Determine the required number of source works by cluster-aware power analysis for the smallest practically meaningful improvement, accounting for paired cluster variance and multiplicity across primary outcomes.
6. Report all property/factor strata and coverage. Do not treat the 30-cluster minimum as a power calculation or a universal sample-size guarantee.

A confidence interval crossing zero does not establish equivalence. A positive interval on one metric does not establish overall superiority if another hard-gated metric materially regresses.
