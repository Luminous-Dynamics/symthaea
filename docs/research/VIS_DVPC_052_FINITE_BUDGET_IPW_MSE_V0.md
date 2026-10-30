# VIS-DVPC-052 — Finite-budget IPW MSE and support truncation v0

**Status:** analytic known-answer benchmark. Support-truncated designs are oracle diagnostics, never identified value estimates or production authorization.

## Research motivation

VIS-DVPC-051 (#9068 / PR #9070) compares logger designs under a fixed positive action-propensity floor. The next question is whether that restriction hides an important finite-budget tradeoff.

Douglas, Persson & Provost, *Logging Policy Design for Off-Policy Evaluation* (arXiv:2605.15108, May 2026), explicitly decompose IPW mean-squared error into squared support-induced bias and variance. Their discussion notes that intentionally omitting some target actions can lower finite-sample MSE in some small-sample or large-action regimes, while the resulting estimator is biased and does not identify the full target-policy value. They further separate overlap (positive mass on target actions) from full support over all actions. See [Sections 3.1–3.3](https://arxiv.org/html/2605.15108) and [Proposition 1](https://arxiv.org/html/2605.15108).

This benchmark isolates that phenomenon in a small exact two-context, two-action Bernoulli DGP. It does not use lower MSE to justify an unsupported policy-value claim.

## Exact finite-sample objective

For target policy pi, logger mu, context distribution P(X=x)=1/2, and per-event IPW score Z=w_pi(X,A)R, define

\`m = E_mu[Z] = sum_x P(x) sum_{a:mu(a|x)>0} pi(a|x) r(x,a)\`.

The bias relative to the full target policy value V(pi) is m - V(pi), exactly the target-reward mass omitted by zero-support actions with a negative sign.

For Bernoulli rewards, R^2=R, so

\`E_mu[Z^2] = sum_x P(x) sum_{a:mu(a|x)>0} pi(a|x)^2 * r(x,a) / mu(a|x)\`.

For N independent logged events, the exact MSE of the sample-mean IPW estimator is

\`MSE_N = (m - V(pi))^2 + (E[Z^2] - m^2)/N\`.

The benchmark computes this per target policy and minimizes the maximum MSE across the candidate portfolio. It does not substitute an ESS proxy or an interval width for MSE.

## Logger-design families

Source: \`tools/vision-dvpc-052-finite-budget-ipw-mse.rs\`.

Two optimization domains are compared under the same target policies, reward world, sample size and grid:

1. **Full-support design:** each action probability is constrained to [0.02, 0.98] in every context. It preserves full target support and the exact IPW estimator is unbiased.
2. **Support-truncating oracle design:** each context may use probability 0 or 1 (omit one action entirely), or a positive interior probability from the same [0.02, 0.98] grid. The design may accept bias when it lowers exact finite-N worst-policy MSE.

The second arm uses exact synthetic reward means to optimize the finite-N MSE. It is intentionally oracle-only. It is not a practical design rule when reward means are unknown; no estimator of those means, confidence bound, or deployment authorization is implied.

The panel uses portfolio sizes M={4,8,32}, balanced/action-1-heavy/action-0-heavy target portfolios, and sample sizes N={1,4,16,64,256,1024}. The full grid has 100 intervals across the positive-propensity region. Smoke uses 12 intervals.

## Mandatory support semantics

Every result reports:
- worst-policy MSE, squared bias, and variance/N separately;
- logger immediate reward and propensity vector;
- omitted target action mass for the worst-MSE policy;
- count of fully identified target policies and whether the entire portfolio is identified.

When any target policy places positive mass on an action with zero logging probability, the corresponding full policy value is **NOT IDENTIFIED** from those logs. The support-truncating arm emits an MSE diagnostic only; it does not emit an identified estimate, confidence interval, best-policy certificate, or value authorization.

A lower oracle MSE is therefore not a victory over support constraints. It is evidence that finite-sample estimation risk has a bias–variance frontier and that the frontier must be reported together with identifiability.

## Known-answer controls

The Rust tests check:
- independent one-policy enumeration of expected IPW score, full target value, squared bias, variance/N, MSE, and omitted target mass;
- zero bias for every registered target policy under a fully supported logger;
- a small-sample control (balanced M=4, N=1) where support truncation lowers the oracle worst-policy MSE while the portfolio is explicitly not identified;
- a larger-sample control (balanced M=4, N=1024) where the unrestricted design selects a full-support optimum;
- deterministic output and exact row/configuration counts;
- invalid sample sizes, probability values, and propensity floors fail closed.

The expected one-policy fixture uses target p1=[0.8,0.2], logger p1=[1,0], N=4:
- expected IPW score: 0.74;
- full target value: 0.765;
- squared bias: 0.000625;
- variance/N: 0.0111;
- MSE: 0.011725;
- omitted target action mass: 0.20;
- identified: false.

These values come from the exact frozen Bernoulli reward table and the finite-N formula above, not Monte Carlo estimates.

## Qualification and claim boundaries

Compile and run the dependency-free harness as follows:

    rustc --edition=2024 -O --test tools/vision-dvpc-052-finite-budget-ipw-mse.rs -o /tmp/vis-dvpc-052-tests
    /tmp/vis-dvpc-052-tests
    rustc --edition=2024 -O tools/vision-dvpc-052-finite-budget-ipw-mse.rs -o /tmp/vis-dvpc-052
    /tmp/vis-dvpc-052 --smoke
    /tmp/vis-dvpc-052

The focused exact-head workflow checks all tests, deterministic replay, 108 result rows, finite MSE decomposition, the small-N non-identification control, the large-N full-support control, and SHA-256 receipts.

The support-truncating result is never presented as identified or safe. The experiment is a finite synthetic risk-frontier study, not a universal policy-design theorem, a SOTA claim, or a license to use biased estimators in production.
