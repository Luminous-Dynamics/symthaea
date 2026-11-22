# MACE Portfolio Interaction Benchmark v0.1

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

Training and calibration use distinct cluster IDs, source-root IDs and task-family IDs. Held-out candidate records also carry source-root IDs, and the candidate selector rejects a test root reused from either training or calibration. Held-out stable-regime tasks use another family and distinct clusters/roots. A shifted regime changes the underlying pair effects; the pairwise selector must abstain and fall back to the additive arm. Unknown relation features also force abstention rather than assuming zero interaction.

The candidate model fits a mean interaction per relation class. It rejects duplicate cluster/relation observations, zero or reused source roots, missing per-cluster relation support, and mixed task-family identity within a cluster. It calibrates an empirical residual envelope on the calibration split, then uses the lower bound (mean minus radius) in selection. The selector rechecks the fitted-model digest before applying it. This is intentionally conservative but **not** a statistical confidence interval or proof of coverage. Held-out evaluation reports false-synergy and false-redundancy counts.

## Metrics

Report independently:
- downstream utility and regret against the evaluator oracle;
- paired utility lift over additive exact with a cluster-bootstrap 95% interval;
- actual redundant-value loss retained;
- actual complementary synergy captured;
- false synergy: predicted positive lower bound but realized interaction is non-positive;
- false redundancy: predicted negative upper bound but realized interaction is non-negative;
- pairwise abstention rate after unknown features or regime shift;
- selected items, context-token spend and worst-case regret.

Governance and provenance failures are hard gates, not terms in a weighted utility score.

## Protocol and split

Five deterministic seeds are run. For each seed:
- 96 training clusters across task families 1 and 2;
- 48 calibration clusters from task family 3;
- 64 held-out stable-regime clusters from task family 4;
- 32 shifted-regime negative-control clusters from task family 5.

The relation model is fitted only on training outcomes; its empirical radius uses calibration outcomes; no held-out outcome is passed into the candidate selector. Cluster, source-root and task-family identities are disjoint by construction.

## Qualification targets (not results)

1. Exact agreement with the evaluator oracle on all tiny enumerated fixture worlds when supplied oracle interactions, as a unit-test control.
2. Positive paired held-out lift over additive exact with cluster-bootstrap 95% lower bound above zero.
3. At most 5% average selection regret versus the held-out oracle on the stable regime.
4. Zero pairwise-model activation on the deliberately shifted regime or unknown relation features.
5. Publish false-synergy and false-redundancy rates, even if the pairwise policy loses to additive.
6. At least five independent seeds, task-family-disjoint evaluation and source-root-disjoint data before any external claim.

## Limitations and next tranche

This v0.1 diagnostic uses a small, explicit relationship vocabulary and a synthetic relation-feature generator. It does not yet estimate semantic overlap from embeddings, learn complementary sets larger than pairs, handle higher-order interactions, or replace the exact additive production portfolio. The empirical calibration radius is an observed residual envelope, not a probabilistic guarantee. No benchmark leaderboard result is implied.

Next:
1. Connect candidate interaction features to the real decision-capsule/evidence graph, not item IDs or opaque labels.
2. Add exact option-level interaction assessments with a receipt binding model, target model, option set, task-family split, source-root split, scope, as-known snapshot and authority.
3. Compare pairwise selection with sampled marginal/Shapley estimates on bounded small portfolios.
4. Test higher-order interactions, diminishing returns, complementarity cycles and adversarial feature misclassification.
5. Evaluate interaction-aware selection only after the baseline additive portfolio tests remain unchanged.


## Source verification status

The implementation contains **13 authored Rust unit tests** by source count. These tests have not been executed in the current environment: `cargo`, `rustc` and `rustfmt` are unavailable. No compile, test, formatting, emitted-output, CI or benchmark run pass is claimed. The authored count is not an executed-test count.
