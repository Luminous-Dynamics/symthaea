# Broca risk-bounded retrieval reference

This is a small, standalone Rust reference crate for evaluating a frozen selective-retrieval policy against a preregistered unsafe-result label. It is deliberately separate from production Symthaea crates and has no third-party dependencies.

## Run the focused tests

From the repository root:

```sh
cargo test --manifest-path research/broca-risk-bounded/Cargo.toml
```

The tests cover the exact zero-failure binomial bound, a known nonzero-failure case, monotonicity, invalid counts/configurations, an empty accepted set, minimum evidence, a fail-closed risk decision, tie-preserving risk-coverage curves, partial-label bounds, and fixed-sequence threshold stopping.

The source is committed but has not been compiled or executed in the authoring environment. Do not treat authored tests as passing until this exact source SHA is built and tested.

## Meaning of the risk gate

The evaluator defines unsafe before the locked test run, for example: a query selected by the frozen approximate policy has strict exact-ID Recall@K below a preregistered threshold, returns a stale fact, or fails a required-evidence check. The exact failure label and denominator must be written in the experiment protocol.

`RiskBudget::assess(unsafe_count, accepted_count)` reports:
- the observed unsafe rate;
- the one-sided exact Clopper-Pearson upper confidence bound;
- a final decision: `Eligible`, `InsufficientEvidence`, or `RiskBudgetExceeded`.

There is no positive decision when zero cases were accepted or when fewer than the configured minimum independent accepted cases exist.

## Why 299 matters

For zero unsafe outcomes in (n) independent accepted cases, the one-sided exact upper bound at confidence level (1-\alpha) is:

```text
p_upper = 1 - alpha^(1/n)
```

At 95% confidence, 299 independent accepted cases with zero observed failures are needed for the bound to fall to at most 1%. With 298, it remains just above 1%. A thousand repeated timing rounds on the same ten queries do not create ten thousand independent reliability observations.

## Statistical limits

This is a simple binomial reference, not a conformal or distribution-free guarantee under arbitrary distribution shift. Its interpretation requires the evaluated query units to be independent or exchangeable under a declared sampling design, with a frozen policy and frozen unsafe-label definition. Split by user/session/conversation or memory lineage where appropriate; repeated queries from the same lineage can be correlated.

The bound concerns the held-out accepted population. It does not estimate risk for each individual query, does not calibrate features by itself, and does not prevent selection leakage if the policy or thresholds are tuned on the final test set. It also does not excuse any cross-scope leakage, deleted-memory resurrection, stale-index acceptance, or other zero-tolerance integrity failure.

For temporal dependence, distribution shift, selective label observation, or repeated monitoring over time, compare with suitable risk-control/conformal methods and prospectively designed audits rather than reusing the simple binomial interval outside its assumptions.

## Recommended evaluation sequence

1. Define the unsafe event and sampling unit before examining test labels.
2. Train/tune the routing policy on development data and select thresholds on a separate calibration partition.
3. Freeze policy, index contract, data processing, and unsafe event.
4. Evaluate only once on a locked test set, reporting accepted/unsafe numerators and denominators per query stratum.
5. Report task utility, coverage/acceptance rate, latency, exact-fallback rate, and integrity failures in addition to the risk bound.
6. Keep each important stratum separate. Do not let a high volume of easy queries conceal poor reliability on temporal updates, unanswerable queries, or privacy-sensitive scopes.

## Risk–coverage curve and fixed-sequence certification

The crate also exposes a descriptive development-set risk–coverage curve:

- Add one RiskObservation per unique query, with a finite acceptance score (higher means more eligible for approximate retrieval) and an evaluator-assigned unsafe label.
- The curve starts at zero coverage, then applies every unique observed score from highest to lowest.
- Equal scores enter together; the implementation never splits ties according to row order.
- Each point reports total coverage, labelled coverage, accepted-label completeness, observed risk among labelled cases, and optimistic/pessimistic empirical risk bounds when outcomes are unknown.
- The optimistic/pessimistic values are deterministic partial-label bounds, not confidence intervals. Unknown accepted outcomes prevent certification; they are never silently treated as safe.

For statistical certification, use a fixed, strictly descending threshold sequence through `certify_fixed_threshold_sequence`. Select this sequence before viewing evaluation labels. The procedure tests from the most selective threshold toward greater coverage, continues only while each eligible threshold passes, and stops at the first unknown-label or risk-budget failure. Subsequent thresholds are marked not tested. Thresholds with too few accepted cases are marked insufficient evidence and are not treated as passes.

The curve generated from observed score values is intended for policy exploration on development data. Do not choose a favorable point using the same labels and then claim that its unadjusted confidence bound is an independent test guarantee. Keep policy development, threshold selection/calibration, and final evaluation separate. The fixed-sequence guarantee also depends on the stated binomial sampling assumptions; it is not a distribution-free guarantee under arbitrary drift or correlated query lineages.

The crate is a qualification primitive for PR #8520, not a replacement for PR #8494's exact-vs-approximate benchmark, MIRE memory-integrity fixtures, or end-to-end memory task evaluation.


## Related in-repository prior work and consolidation boundary

The CRISPR selective-risk track already has adjacent research work in [PR #7502](https://github.com/Luminous-Dynamics/symthaea/pull/7502) and [issue #7580](https://github.com/Luminous-Dynamics/symthaea/issues/7580), including risk–coverage curves, context-stratified summaries, and unknown-label-aware error bounds for probability scores. This memory-specific reference is not a claim that risk–coverage analysis or selective prediction is novel.

The distinguishing slice here is retrieval-oriented fixed-sequence certification against a configured unsafe-result rate, tied acceptance-score tiers, unique query-ID enforcement, and a fail-closed stop when accepted outcomes are unlabelled. The present curve/reporting code is deliberately local to this standalone research crate; before production use or a general-purpose release, compare both implementations on shared fixtures and factor shared metric semantics into one maintained implementation rather than allowing parallel score/risk conventions to drift.


## Cross-domain selective-risk contract fixture

[`fixtures/SELECTIVE_RISK_CROSS_DOMAIN_V1.json`](fixtures/SELECTIVE_RISK_CROSS_DOMAIN_V1.json) pins expected outputs for zero coverage, exact score ties, a measured failure, an unknown accepted label, coverage denominators, and best-/worst-case empirical risk. It is intended to be run against both this retrieval evaluator and the adjacent [CRISPR selective-risk kernel in PR #7502](https://github.com/Luminous-Dynamics/symthaea/pull/7502) before either implementation is factored into shared code.

The fixture normalizes an evaluator-specific event into `failure=true|false|null`. That is only a metric adapter: CRISPR classification error and unsafe memory retrieval are different events, and their policies, calibration meaning, contexts, sampling units, and authority boundaries must remain domain-specific. Both adapters must preserve unknown labels, treat score ties atomically, and report an explicit zero-coverage origin with undefined rates as `null`.

The fixture is synthetic. Its JSON and arithmetic require independent checking; it is not a Rust test result, a benchmark result, a safety certificate, or evidence of improved retrieval quality.
