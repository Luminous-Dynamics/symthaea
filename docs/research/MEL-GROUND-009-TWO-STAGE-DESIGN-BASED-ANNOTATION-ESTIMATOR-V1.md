# MEL-GROUND-009 — Two-stage design-based annotation estimator v1

**Status:** implementation draft in PR #13692; exact-head qualification pending  
**Date:** 25 November 2026  
**Upstream sampling authority:** #13626 / #13624  
**Estimator:** `temporal_design_estimator.rs`

## Problem

The acquisition planner deliberately separates two streams:

1. **Active refinement** targets known model errors, human disagreement, coverage gaps and hidden-repeat controls. It is adaptive by design and useful for improving the system.
2. **Probability audit** samples source works and then query items within those works using a frozen design and known inclusion probabilities. It exists to estimate finite-population performance without pooling adaptive selections into the benchmark.

Active selections cannot be treated as a random sample. The estimator therefore replays the planner and consumes only completed outcomes for the exact probability-audit draw.

## Two different estimands

The estimator reports both and does not silently interchange them.

### Query-item-weighted estimand

Every eligible query item receives equal population weight. For item outcome (y_{wi}\in[0,1]) and first-order inclusion probability (\pi_{wi}):

[
\widehat{T}_{item}=\sum_{(w,i)\in s}\frac{y_{wi}}{\pi_{wi}},
\qquad
\widehat{\mu}_{item}=\frac{\widehat{T}_{item}}{\sum_w N_w}.
]

This answers: *How well does the system perform on the finite population of eligible query items?*

### Source-work-weighted estimand

Each source work receives equal target weight, regardless of how many query items it contributes. The source-work outcome is the mean within that work:

[
\bar y_w=\frac{1}{N_w}\sum_{i=1}^{N_w}y_{wi}.
]

The equal-work total and mean are:

[
\widehat{T}_{work}=\sum_{(w,i)\in s}\frac{y_{wi}/N_w}{\pi_{wi}},
\qquad
\widehat{\mu}_{work}=\frac{\widehat{T}_{work}}{W}.
]

This answers: *How well does the system perform on an average source work, rather than an average query?* It prevents works with many annotated excerpts from dominating the target metric.

## Inclusion probabilities

Within each stratum, the design selects (K) source works without replacement from (W) eligible works. Within selected work (w), it selects (m_w) query items without replacement from (N_w) eligible items.

The first-order item inclusion probability is:

[
\pi_{wi}=\frac{K}{W}\frac{m_w}{N_w}.
]

For two distinct items from the same work:

[
\pi_{wi,wj}=\frac{K}{W}
\frac{m_w(m_w-1)}{N_w(N_w-1)}.
]

For items from different works (w\ne v):

[
\pi_{wi,vj}=\frac{K(K-1)}{W(W-1)}
\frac{m_w}{N_w}\frac{m_v}{N_v}.
]

The variance routine uses the general Horvitz–Thompson pairwise form, separately for (y_{wi}) and (y_{wi}/N_w). It uses the actual selected work/query counts, not a constant-size approximation.

## Trust and reproducibility controls

The input must match the frozen metric, model identity, evaluator version and scoring-plan digest. Each outcome row carries the exact item ID, source-work digest, scoring-report digest and matching execution/scoring identity tuple.

Before estimation, the module:
- replays the deterministic acquisition planner from the candidate frame, configuration and seed;
- compares the replayed selection receipt to the expected digest when one is supplied;
- rejects missing, duplicate, extra, wrong-metric, wrong-source, wrong-model, wrong-evaluator, wrong-plan, duplicate-report-receipt and out-of-range outcomes;
- accepts only outcomes for the selected probability-audit stream;
- rejects a stratum with eligible items but zero audit inclusion probability rather than assigning it zero performance;
- uses checked population-count accumulation.

The selection receipt is a deterministic commitment, not a signature or authentication mechanism. The model, evaluator, scoring-plan and report digests are input metadata until an independent verifier authenticates their provenance.

## Variance and insufficiency

The estimator returns variance unavailable when the design cannot estimate it, including:
- fewer than two selected works when multiple eligible works exist;
- insufficient within-work sample support for required pairwise inclusion probabilities;
- zero/invalid joint inclusion probabilities;
- negative or non-finite variance calculations.

It does not turn any of these conditions into zero variance. A full census of a one-work population has zero design variance. Sampling one work from a larger multi-work population does not identify between-work variance.

A design variance is not automatically a population confidence claim. The current report permanently declares `observed_outcomes_authenticated=false` and `population_confidence_claim_permitted=false` until the sampling frame, source lineage, model execution, scoring outputs, rights/consent and audit outcome process are independently qualified.

## Synthetic oracles

Two tests validate distinct parts of the estimator.

1. **One-work complete census:** two queries from one eligible source work, both sampled. The query-item total is 1.0 and mean 0.5; the source-work total and mean are both 0.5. Both design variances are zero because the entire finite population is observed.
2. **Exhaustive three-work oracle:** three works have query-item totals ([1,2,0]), two queries per work. All three possible two-of-three work samples are reached. Across those designs, the average query-item HT total is 3.0 and average estimated query-item total variance is 1.5. The source-work means are ([0.5,1.0,0.0]); the average equal-work HT mean is 0.5 and the average estimated source-work total variance is 0.375.

These are synthetic finite-population correctness tests. They do not measure real Melothaea performance, musician agreement, or generalization to an external music population.

## Competitive significance

This closes an important evaluation gap between active error-driven improvement and defensible benchmark estimates. It enables a future claim such as “the model improved on the frozen source-work probability audit” to be backed by the actual two-stage design, separate item- and work-weighted estimands, and variance diagnostics.

It does not itself improve temporal grounding or prove SOTA. The next empirical requirement is a rights-cleared, source-lineage-disjoint finite population with enough independently sampled works, pre-registered metric/scoring identities, independently authenticated execution/scoring receipts, and an adequate power/sample-size plan.

## Implementation

- PR #13692 — draft: [Add two-stage design-based annotation estimator](https://github.com/Luminous-Dynamics/symthaea/pull/13692)
- Planner: #13626
- Acquisition protocol: #13624
- CI uses Rust 1.96, nested locked Cargo workspace, exact-source tests, warnings-denied Clippy and a CLI replay of the synthetic complete census.
