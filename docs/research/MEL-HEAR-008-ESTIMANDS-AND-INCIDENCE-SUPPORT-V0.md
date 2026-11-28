# MEL-HEAR-008 — Estimands and incidence support for unbalanced performer × work panels v0

**Status:** research contract / deterministic synthetic fixture; not a production estimator  
**Date:** 28 November 2026  
**Parent:** #14026  
**Related:** #14019, #14028, #14030, #10417

## Executive result

The same unbalanced observed panel can support several different descriptive means. It does **not** follow that those means estimate the same population quantity.

This increment freezes the first research contract around a complete 3-work × 3-performer incidence panel with seven observed cells and two explicitly `missing_not_sampled` cells. It deliberately contains no missing-cell imputation, no fitted random-effects model and no confidence interval. Its purpose is to make the weighting decision visible before any bootstrap or small-sample correction is compared.

## Why estimand-first matters

For a balanced complete panel, equal weighting of observed cells coincides with giving each work and performer equal marginal weight. Once the incidence matrix is unbalanced, those targets separate.

Consider the retained synthetic cell outcomes:

| Work | Performer 1 | Performer 2 | Performer 3 |
|---|---:|---:|---:|
| Work 1 | 0 | 0 | missing |
| Work 2 | 1 | missing | 1 |
| Work 3 | 0 | 1 | 1 |

The bipartite incidence graph is connected, but two of nine product cells are explicitly marked `missing_not_sampled`. The supported descriptive estimands use only the seven observed outcomes. The uniform product target assigns 1/9 weight to each of all nine cells, including both missing cells, and is therefore not estimable without an explicit model/transport assumption. Connectivity does not make the missing outcomes observed, nor does it license extrapolation.

## Exact deterministic results

| Estimand | Weighting rule | Exact point estimate |
|---|---|---:|
| Observed-cell mean | Each of 7 observed cells receives 1/7 | 4/7 = 0.571429 |
| Equal-work macro mean | Mean within each work, then average 3 work means | 5/9 = 0.555556 |
| Equal-performer macro mean | Mean within each performer, then average 3 performer means | 11/18 = 0.611111 |
| Uniform product-population mean | Each of 9 work × performer combinations receives 1/9 | **Not estimable** without assumptions for the 2 missing cells |

This is not a performance benchmark or an uncertainty result. It is an executable known-answer fixture showing that "the mean paired effect" is underspecified in an unbalanced panel.

## Contract v0

The research directory contains a retained synthetic panel and a draft JSON Schema for the manifest shape.

The contract records all nine potential single-performer cells, including seven observed cells and two explicit not-sampled records. The contract records:
- synthetic or digest-shaped work, performer, cell and session locators (never raw performer identities);
- the work × performer incidence cells and observed/missing status;
- performer membership weights in the schema; v0 execution accepts only one performer per cell and rejects ensemble/set-valued membership until a distinct ensemble estimand is defined;
- distinct estimand kinds and explicit per-cell weights;
- target work and performer marginals for the declared product-population estimand; the full target weight vector includes both missing cell records;
- connected components, isolated nodes, and maximum row/column degree;
- a fail-closed disposition when target cells are missing and extrapolation is not authorized.

JSON Schema checks shape only. The standalone Rust qualifier at `tools/mel-hear-008-qualifier` additionally validates the explicit W × P incidence grid, membership and estimand-weight sums, graph connectivity, declared support diagnostics, known-answer arithmetic, missing-target refusal and row-order invariance. It still does not authenticate identities, validate sampling probabilities or establish statistical validity.

## Executable v0 qualifier

The independent Rust 1.96 harness in `tools/mel-hear-008-qualifier` parses the fixture with unknown fields rejected, verifies every potential work × performer cell is represented exactly once, recomputes support graph components/degrees, validates the three weighting rules and checks the retained known answers. Negative controls reject omitted cells, duplicate pairs, incorrect support diagnostics, invalid membership weights, unsupported missing-cell claims and altered expected means. A row-order permutation test checks that the report is invariant to manifest order.

The qualifier is deliberately fail-closed on ensemble/multi-performer rows. Although the draft schema can represent membership sets, v0 does not pretend that one ensemble observation is several independent performer observations. A future extension needs an explicit group-level or membership-weighted estimand before those records are supported.

These checks establish structural/arithmetic contract behavior only; they do not estimate uncertainty or select a production inference method.

## Statistical consequences

Candidate procedures must be compared against the **same estimand**:
- observed-cell resampling answers a sampled-cell question unless the sampling design justifies a broader target;
- work-marginal inference must preserve the equal-work weighting rule;
- performer-marginal inference must preserve the equal-performer weighting rule;
- a product-population target with absent cells needs a declared model/transport assumption or a not-estimable disposition;
- multi-performer cells require an explicit ensemble or multiple-membership estimand. Never duplicate one ensemble cell once for every member and treat those copies as independent observations.

The next simulation increment under #14026 should hold the data-generating process fixed while varying incidence imbalance, degree concentration, missingness mechanism and graph connectivity. Each candidate method must report bias, interval coverage, type-I error, interval width, Monte Carlo uncertainty and rejection/not-estimable rate against the same declared target.

## Fail-closed rules

1. Missing or ambiguous estimand/weights: no population interval.
2. Unobserved target cells with no declared extrapolation model: not estimable for that target.
3. Disconnected incidence components: no across-component population claim without explicit identifying assumptions.
4. v0 qualifier rejects all set-valued/ensemble cells; future membership weights must sum to one and be tied to a declared group or membership estimand.
5. Report sample counts and effective support separately; neither a raw count nor a connected graph is a power calculation.
6. No automatic item-wise bootstrap fallback.

## Claim ceiling

This fixture establishes only the arithmetic distinction among three observed-panel means and the non-estimability of a uniform product target under the stated no-extrapolation rule. It does not qualify any inferential method, validate real performer independence, authenticate audio or annotations, establish cohort representativeness, or demonstrate Melothaea model quality.
