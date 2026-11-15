# Fusion rejected-candidate audit v0.2

A dependency-free Rust contract for statistically bounding missed feasible candidates when exhaustive high-fidelity evaluation of every screened-out design is impractical.

This is a synthetic/research contract. It is not a plasma model, solver adapter, or physical validation method. The full-census reduced-space benchmark in [fusion-search-benchmark-v0-1](../fusion-search-benchmark-v0-1/) remains the reference for checking whether this audit bound behaves conservatively.

## Paradigm

A screening optimizer must retain every candidate and its disposition. When a full reference evaluation of every rejected candidate is too expensive, draw an independently specified, reproducible, stratified random sample from the rejected population.

For each stratum, the audit inverts a one-sided **exact finite-population hypergeometric test** to obtain an upper confidence bound on the number of candidates that are either:

- confirmed feasible by the declared reference evaluation; or
- unresolved because the reference evaluation failed, did not converge, was unsupported, or was inconclusive.

Unresolved outcomes are conservatively treated as potentially feasible. The per-stratum error budget is allocated with a Bonferroni correction, producing a simultaneous family-wise confidence bound across strata. A full census returns the exact observed potential-feasible count.

The recall lower bound uses a complete accepted-population census, not only a list of accepted successes:

`recall_lower = A_feasible / (A_feasible + A_unresolved + sum(stratum_missed_upper))`

Here, A_feasible is the count of reference-confirmed feasible candidates among all accepted candidates. A_unresolved counts every accepted candidate whose reference result is failed, non-converged, unsupported or inconclusive. Such accepted outcomes are included in the denominator but not in the lower-bound numerator. Rejected strata contribute their simultaneous upper bounds. Confirmed-infeasible accepted candidates contribute to neither term.

This corrects a dangerous failure mode: reporting a high recall lower bound after omitting accepted-but-unresolved candidates from the denominator.

## Contract and invariants

- The frozen universe is a complete, unique, lexicographically ordered candidate identity list.
- The accepted population is a complete reference-outcome census, including confirmed infeasible and unresolved results.
- The accepted identities and rejected strata partition the frozen universe exactly once.
- Each rejected stratum has a frozen identity list, definition digest, population digest, explicit sample size, seed-derived selection receipt and per-outcome evidence references.
- Samples are uniform without replacement under the declared deterministic draw; the module verifies every sampled identity in exact draw order.
- An unbiased bounded-integer draw is used inside partial Fisher–Yates selection.
- The inclusion probability for each candidate in a stratum is n/N.
- Every declared sample slot must have exactly one corresponding outcome. Missing, duplicate, unexpected or tampered sample identities are rejected.
- Only ConfirmedInfeasible counts as a negative observation. All other outcomes are conservatively counted as potential feasible outcomes.
- Exact one-sided hypergeometric inversion produces a stratum upper bound. Bonferroni allocation provides simultaneous family-wise confidence without requiring independence between stratum tests.
- Recall certification is refused when any population identity is missing, duplicated or outside the frozen partition.

## Statistical method

For a stratum with population size N, without-replacement sample size n, and x observed potential-feasible outcomes, let X follow Hypergeometric(N, K, n), where K is the unknown number of potential-feasible candidates in that finite stratum.

For per-stratum error probability alpha_s, the upper count U is the largest K not rejected by the one-sided lower-tail test:

`P[X <= x | N, K, n] > alpha_s`

With S non-empty strata, alpha_s = (1 - familywise_confidence_level) / S. The total rejected upper bound is the sum of the stratum upper counts. This is a finite-population bound, not a binomial approximation and not a model-based extrapolation.

## Numerical implementation note

The hypergeometric tail is evaluated in the log domain using IEEE-754 `f64`, log-factorials and log-sum-exp, then inverted by monotone binary search over the finite population success count. It is the exact finite-population hypergeometric test construction, not a binomial/Hoeffding approximation; the numerical probabilities themselves are floating-point evaluations, not exact rational arithmetic. The test suite exhaustively enumerates coverage for every population size through 18, every sample size and every possible true success count at nominal 95% confidence, plus census-collapse and monotonicity checks. Very large populations or decisions near a regulatory confidence threshold should receive an independent high-precision numerical cross-check before the result is used as engineering evidence.

## Sample-selection integrity

The crate uses a versioned deterministic SplitMix64 / partial Fisher–Yates selection algorithm. It derives the expected sample identities from the frozen, lexicographically ordered rejected-candidate identities, seed, and stratum identity. The supplied sample identities must match the deterministic draw in exact order.

SplitMix64 is **not cryptographically secure**. The seed commitment, population manifest, selection receipt and evidence references are declared identities; this crate does not fetch or hash their payloads, authenticate timestamps, prove that the seed was committed first, verify independent execution, or establish solver-label truth. A production audit must obtain an unpredictable seed from an external cryptographically secure randomness source after the population and strata are frozen, commit/reveal it under independent timestamped control, and independently verify source bytes, sample selection and solver evidence.


## Conditional decision-opportunity audit

The follow-on `decision_opportunity` module addresses a separate question: can a screen preserve *designs that could change the engineering decision*, not merely designs that are feasible?

It reuses the frozen candidate universe, accepted/rejected partition, sample identities, exact hypergeometric inversion and Bonferroni familywise correction above. It does not implement a second sampler or statistical engine.

The decision audit freezes:
- the full objective vector, units, directions and per-objective materiality tolerances;
- the incumbent reference frontier and its identity;
- the opportunity rule and campaign identity;
- a complete reference evaluation for every accepted candidate;
- one reference evaluation for every sampled rejected candidate, in the exact frozen sample draw order.

The v0.1 rule identifies a **conditional reference-front opportunity** when a candidate is confirmed feasible, lies within the declared model domain, is not epsilon-dominated or epsilon-equivalent to an incumbent, and materially improves at least one objective against the frozen frontier. A candidate may replace or improve an incumbent; this is not global Pareto-front discovery.

The report gives two distinct lower bounds:
- feasible-candidate recall, inherited from the original reject audit;
- conditional reference-front opportunity recall, evaluated against the frozen incumbent frontier and objective manifest.

Out-of-domain, failed, non-converged, unsupported and inconclusive evaluations remain unresolved and count as potential missed opportunities in the conservative rejected-population bound. A missing sample result, changed candidate order, altered objective identity or changed frontier is refused or requires a new campaign.

The report preserves the frozen universe, accepted-population, objective, opportunity-rule, incumbent-frontier, stratum-population and sample-selection receipt identities. This is essential: the opportunity result is conditional on these exact inputs.

### Scope limits

- The opportunity recall is conditional on the frozen incumbent reference set. It is **not global Pareto recall** and cannot establish that an unenumerated physical design space has been searched adequately.
- The epsilon rule is a frozen benchmark definition, not an industry standard. Objective units, direction, scale and tolerances must be justified and fixed before outcomes are viewed.
- No average decision regret is emitted. A bounded regret metric requires finite, defensible objective bounds and a separate conservative derivation.
- The objective evaluator consumes exact declared reference outcomes; it does not launch PROCESS or another physical solver.
- Synthetic fixtures test the contract only. They are not physical validation or reactor-design evidence.

The module adds 12 authored Rust unit tests for frontier extension, the feasibility/opportunity recall distinction, hard/inconclusive/out-of-domain results, epsilon dominance/equivalence, vector units/order, missing/retained sample identities, baseline frontier consistency and zero-sample conservatism. Across the original 15 rejected-candidate-audit tests and these 12 new authored tests, the crate contains 27 authored Rust tests. This is a source inventory, not a claim that the new tests have executed.

## Run

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path Cargo.toml --bin audit_rejected_candidates
cargo run --manifest-path Cargo.toml --bin audit_decision_opportunities
```

## Claim ceiling

A PASS means only that the declared sample identities match the deterministic selection algorithm, the accepted/rejected partition is complete, unresolved outcomes are handled conservatively, and the exact hypergeometric bounds and recall arithmetic behave as tested on synthetic cases.

It does not prove sampling independence, source-byte integrity, label truth, solver correctness, physical feasibility, plasma performance, reactor economics, safety, or a real-world 99% recall claim. A measured solver-backed audit must be independently reproduced before the bound can support an engineering decision.
