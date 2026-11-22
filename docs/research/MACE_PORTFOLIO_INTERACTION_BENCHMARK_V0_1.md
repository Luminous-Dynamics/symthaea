# MACE Portfolio Interaction Benchmark v0.3

**Snapshot:** 2026-11-22  
**Status:** Synthetic benchmark implementation and protocol; no measured result or SOTA claim.

## Research question

Does selecting memory as a set, using estimated redundancy and complementarity, improve held-out downstream utility over exact additive selection under identical resource constraints?

An option's value is conditional on the rest of the portfolio:

`marginal_value(e | S) = U(S ∪ {e}) - U(S)`

The sign and size of this marginal value can change with the selected set. Memory portfolios may contain redundant paraphrases, complementary fragments, mutually conflicting procedures, independent knowledge, or alternative representations of the same event. Additivity must be measured, not assumed.

## Implemented diagnostic

The evaluator binary `mace_portfolio_interactions` compares:
- **Additive exact:** exhaustive subset selection using the same singleton-value predictions and token budget, but no interaction terms.
- **Pairwise conservative:** exhaustive subset selection using the same singleton predictions plus the calibrated lower bound of each pairwise interaction estimate.
- **Evaluator oracle:** exhaustive subset selection using hidden realized singleton values and realized pair interactions. This is an upper-bound diagnostic only and is not available to the candidate.

The current synthetic world has six candidate items, a four-token budget, and one-representation-per-event constraints. Interaction classes:
- redundant: selecting both produces negative incremental value;
- complementary: selecting both produces positive synergy;
- independent: interaction is zero;
- conflicting: selecting both incurs a penalty.

Training and calibration use distinct cluster IDs, source-root IDs and task-family IDs. The same-family conformal holdout shares task family 3 with calibration but uses disjoint cluster/root identities and the same stable interaction/noise generator. Stable task-family-4 and shifted task-family-5 examples are reported as separate out-of-distribution diagnostics. Candidate roots are checked against training and calibration roots. A shifted regime changes the underlying pair effects; the pairwise selector must abstain and fall back to the additive arm. Unknown relation features also force abstention rather than assuming zero interaction.

The candidate model fits a mean interaction per relation class. It rejects duplicate cluster/relation observations, zero or reused source roots, missing per-cluster relation support, and mixed task-family identity within a cluster. The radius is the 90% split-conformal absolute-residual quantile using rank `ceil((n_calibration + 1) × 0.90)`; selection uses the resulting lower bound. The selector rechecks the fitted-model digest before applying it. For the task-family-3 conformal holdout, calibration and test residuals use the same stable relation/noise generator with disjoint clusters and roots; under this exchangeability premise, the quantile targets finite-sample marginal prediction-interval coverage for each true relation class. Coverage is evaluated using evaluator-owned relation classes, independently of candidate relation-label errors and unknown labels. These are prediction intervals for pair effects, not confidence intervals for the mean. The exchangeability premise is not extended to task-family-4 or the shifted task-family-5 holdout. Held-out evaluation reports false-synergy and false-redundancy counts.

## Metrics

Report independently:
- downstream utility and regret against the evaluator oracle;
- paired utility lift over additive exact with a cluster-bootstrap 95% interval;
- actual redundant-value loss retained;
- actual complementary synergy captured;
- false synergy: predicted positive lower bound but realized interaction is non-positive;
- false redundancy: predicted negative upper bound but realized interaction is non-negative;
- empirical prediction-interval coverage, reported separately for correctly classified stable and shifted relation features;
- relation-feature misclassification and unknown-feature counts, so interval coverage is never conflated with classifier accuracy;
- pairwise abstention rate after unknown features or regime shift;
- selected items, context-token spend and worst-case regret.

Governance and provenance failures are hard gates, not terms in a weighted utility score.

## Protocol and split

Five deterministic seeds are run. For each seed:
- 96 training clusters across task families 1 and 2;
- 48 calibration clusters from task family 3;
- 64 stable-regime conformal-evaluation clusters in task family 3, with cluster and source-root identities disjoint from calibration;
- 32 stable-regime out-of-family clusters in task family 4;
- 32 shifted-regime negative-control clusters in task family 5.

The task-family-3 conformal holdout uses the same stable relation-generation distribution and the same bounded interaction-noise distribution (uniform integer perturbation in [-2, 2]) as calibration, but fresh clusters and roots; it is the only split for which the exchangeability premise is intended. Task-family-4 and shifted task-family-5 are separate out-of-distribution diagnostics.

The relation model is fitted only on training outcomes; its split-conformal residual quantile uses calibration outcomes; no held-out outcome is passed into candidate selection. Interval coverage is reported separately for the same-family conformal holdout, stable task-family OOD and shifted-regime holdout, and against evaluator-owned true relation classes, independently of candidate relation labels. Relation-label errors and unknown labels are counted as separate feature-quality diagnostics. Training/calibration clusters and roots are disjoint; calibration/test cluster and root IDs are also disjoint.


## Qualification targets (not results)

1. Exact agreement with the evaluator oracle on all tiny enumerated fixture worlds when supplied oracle interactions, as a unit-test control.
2. Positive paired held-out lift over additive exact with cluster-bootstrap 95% lower bound above zero.
3. At most 5% average selection regret versus the held-out oracle on the stable regime.
4. Zero pairwise-model activation on the deliberately shifted regime or unknown relation features.
5. Publish false-synergy and false-redundancy rates, even if the pairwise policy loses to additive.
6. At least five independent seeds, task-family-disjoint evaluation and source-root-disjoint data before any external claim.

## Limitations and next tranche

This v0.1 diagnostic uses a small, explicit relationship vocabulary and a synthetic relation-feature generator. It does not yet estimate semantic overlap from embeddings, learn complementary sets larger than pairs, handle higher-order interactions, or replace the exact additive production portfolio. For the task-family-3 conformal holdout, calibration and test residuals are generated from the same stable relation distribution with disjoint clusters and roots; under this exchangeability premise, the quantile targets finite-sample marginal prediction-interval coverage for each true relation class. It is not a confidence interval for the mean. The premise is not extended to task-family-4 or regime-shifted task-family-5 evaluation. No benchmark leaderboard result is implied.

Next:
1. Connect candidate interaction features to the real decision-capsule/evidence graph, not item IDs or opaque labels.
2. Add exact option-level interaction assessments with a receipt binding model, target model, option set, task-family split, source-root split, scope, as-known snapshot and authority.
3. Compare pairwise selection with sampled marginal/Shapley estimates on bounded small portfolios.
4. Test higher-order interactions, diminishing returns, complementarity cycles and adversarial feature misclassification.
5. Evaluate interaction-aware selection only after the baseline additive portfolio tests remain unchanged.


## Source verification status

The implementation contains **17 authored Rust unit tests** by source count. These tests have not been executed in the current environment: `cargo`, `rustc` and `rustfmt` are unavailable. No compile, test, formatting, emitted-output, CI or benchmark run pass is claimed. The authored count is not an executed-test count.
