# AR-5B28 — Vector Confidence-Set Bake-off v0.1

**Status:** research/benchmark protocol; theorem replication required before Rust implementation.  
**Tracking:** [Issue #10009](https://github.com/Luminous-Dynamics/symthaea/issues/10009)  
**Branch:** **research/ar5b28-vector-confidence-bakeoff-v0.1-20261102**

## 1. Why vector confidence sets are the next paradigm shift

The current scalar Hoeffding + Bonferroni approach constructs a separate uncertainty radius for each registered policy/metric row. It is simple and auditable, but it discards within-policy covariance and correlation between guardrail metrics.

A policy arm often produces a joint vector:

\[
X_i=(\text{crash risk},\text{access},\text{delay},\text{cost},\text{quality},\ldots)\in\mathbb R^d.
\]

The policy optimizer does not need every marginal interval independently. It needs support bounds in the exact registered linear guardrail directions. A joint confidence ellipsoid can yield:

\[
\sup_{\mu\in E} c^\top\mu
=c^\top\bar X+r\sqrt{c^\top Bc},
\qquad
\inf_{\mu\in E} c^\top\mu
=c^\top\bar X-r\sqrt{c^\top Bc},
\]

where \(E=\{\mu:(\mu-\bar X)^\top B^{-1}(\mu-\bar X)\le r^2\}\).

For any fixed ellipsoid event that covers the full metric vector, these bounds hold simultaneously for every direction \(c\), not merely a preselected scalar list. In AR-5B26, an upper guardrail can use the support value and a lower guardrail the infimum. For policy mixtures, construct a simultaneous vector set per policy arm and allocate the arm-level failure budget across the frozen policy family; then combine the per-arm directional support bounds linearly in \(q\). The policy × metric × stratum roster must be frozen before observing outcomes.

This could reduce conservatism without weakening coverage. It is a hypothesis to benchmark, not a result already achieved.

## 2. Candidate theorem A — bounded-vector empirical Bernstein

Martinez-Taboada & Ramdas, “Empirical Bernstein in Smooth Banach Spaces,” *Annals of Applied Probability* 36(3), 2199–2236 (June 2026), DOI 10.1214/25-AAP2277:

- Bounded random vectors in 2-smooth separable Banach spaces, including finite-dimensional Euclidean spaces and separable Hilbert spaces.
- Fixed-sample and stopping-time confidence sets.
- Variance-adaptive width with a leading term matching Bernstein asymptotically.

The exact theorem, norm smoothness constant, variance estimator, filtration, stopping-time statement, and finite-sample constants must be transcribed from the paper and supplement. Do not assume an arbitrary sample covariance ellipsoid inherits its coverage theorem.

## 3. Candidate theorem B — matrix-heat covariance-adaptive ellipsoid

A very recent comparator is Ramdas & Martinez-Taboada, “Variance-adaptive Chernoff bounds: nonasymptotically valid, asymptotically sharp confidence sets for unbounded means,” arXiv:2610.42058v1, posted 30 October 2026 (PDF dated 2 November 2026):
https://arxiv.org/abs/2610.42058

### 3.1 Exact model

For vector observations \(X_i\in\mathbb R^d\), the Section 4 model is a common conditional mean \(\mu\) with a known directional conditional sub-Gaussian proxy:

\[
\mathbb E[\exp\{u^\top(X_i-\mu)\}\mid\mathcal F_{i-1}]
\le \exp\{s^2\|u\|_2^2/2\},\qquad u\in\mathbb R^d.
\]

Finite-sample coverage is under this conditional model. The fixed-dimensional IID law with positive-definite covariance \(\Sigma\) is used for the asymptotic true-covariance chi-square efficiency conclusion. The positive-definiteness assumption is not needed for finite-sample coverage.

Theorem 4.1 gives a matrix-heat ellipsoid with conditional-on-pilot miscoverage at most \(b\). At a fixed IID law with \(\Sigma\succ0\), its rescaled ellipsoid converges to the true-covariance Gaussian chi-square geometry. It is not a generic distribution-free result for changing conditional means.

### 3.2 Pilot/scoring split and calibration recipe

For a planned fixed sample size \(n\ge4\), the paper uses:

\[
k=\max\{2,\lfloor n^{2/5}\rfloor\},\qquad N=n-k.
\]

The pilot covariance is computed from the first \(k\) observations and its eigenvalues are clipped to:

\[
[s^2/\log(n+e),\,s^2]
\]

to obtain a positive-definite pilot matrix \(B\). The scoring sample yields its mean \(z\) and scatter \(Q\). No pilot accuracy guarantee is needed for coverage; the pilot is conditioned upon. For \(n<4\), use the paper's valid normal-mixture ball fallback.

For the matrix-heat method, Appendix D.3 sets:

\[
\rho=\frac{1}{8\log(n+e)},\qquad
W_G=s\sqrt{\frac{d}{2bN\rho}},
\]

\[
Q_B=B^{-1/2}QB^{-1/2},\qquad
M_B=\frac{4d^{3/2}s^3}{\lambda_{\min}(B)^{3/2}},
\]

\[
\tau=\left(\frac{4M_BN}{b\rho(1-2\rho)}\right)^{2/3},
\qquad
A=N(1+8\rho)+\frac{NW_G^2}{\lambda_{\min}(B)}+2\tau.
\]

The scoring covariance clock uses guarded updates: accept an update only while the remaining covariance stays above \(\tau I\); freeze on the prescribed eigenvalue condition. The observable audit is:

\[
\lambda_{\max}(Q_B)\le N(1+8\rho).
\]

If it passes, use the matrix-heat ellipsoid and the continuation probability/cutoff in Section 4.1, equation (4.6). If it fails, return the conservative radius-\(W_G\) ball. The fallback is part of the theorem-backed procedure, not an optional implementation convenience.

The implementation gate must transcribe equation (4.4), equation (4.5), equation (4.6), Appendix D.3 and their exact constants from the PDF; the abbreviated formulas above are not a substitute for the full proof or algorithm.

### 3.3 What this means for policy guardrails

For a vector of registered policy outcomes with coordinate ranges \(R_j=b_j-a_j\), Hoeffding's lemma gives a valid directional sub-Gaussian proxy:

\[
s=\frac{1}{2}\sqrt{\sum_{j=1}^d R_j^2}.
\]

This follows because the range of \(u^\top X\) is at most \(\sum_j|u_j|R_j\le\|u\|_2\|R\|_2\). For \(d\) coordinates all in \([0,1]\), this simple proxy is \(s=\sqrt d/2\). It may be conservative, but the paper's covariance adaptation may recover direction-specific precision.

For each policy arm, the same ellipsoid event must cover the full frozen vector of guardrail metrics. A registered guardrail direction \(c_g\) then gets a support bound:

\[
U_{ig}=c_g^\top z_i+r_i\sqrt{c_g^\top B_ic_g},\qquad
L_{ig}=c_g^\top z_i-r_i\sqrt{c_g^\top B_ic_g}.
\]

Only if those bounds share a simultaneous coverage event may AR-5B26 safely use:

\[
\sum_i q_iU_{ig}\le\tau_g,\qquad
\sum_i q_iL_{ig}\ge\tau_g.
\]

Do not create a separate vector ellipsoid for every guardrail direction and then forget the policy-family confidence allocation. The ellipsoid covers all directions for one arm; the union across policy arms and other independent evidence families still has to be charged to the error budget.

### 3.4 Crucial model distinctions

- The matrix-heat ellipsoid is a **fixed-time vector-mean** construction.
- The paper's scalar confidence sequence is a **time-uniform scalar-mean** construction with an LIL-optimal leading constant.
- Its linear-bandit vector sequence uses a different score model with predictable bounded actions and conditionally sub-Gaussian residual noise.
- Do not relabel the scalar confidence sequence or the linear-bandit result as a time-uniform vector-mean confidence ellipsoid.
- The scalar heavy-tail extension under a known conditional absolute central moment \(p>2\) is not automatically a multivariate heavy-tail theorem.

## 4. Benchmarks: matched coverage, not just smaller intervals

Compare the following methods on exactly the same frozen vectors, sample splits, confidence failure allocations, support rosters and data seeds:

1. Existing scalar one-sided Hoeffding + Bonferroni.
2. Bounded-vector empirical Bernstein in smooth Banach spaces (Candidate A).
3. Matrix-heat vector ellipsoid (Candidate B).
4. The paper's capped sample-scatter mixture ellipsoid (Appendix D.2).
5. Horizon-optimized isotropic normal-mixture ball (Appendix D.1).
6. A coordinate-wise vector Hoeffding/Bonferroni baseline.
7. Known-covariance Gaussian ellipsoid as a diagnostic oracle only; never a deployable baseline.

### Frozen data regimes

- IID Gaussian vectors with diagonal, equicorrelated, and adversarial covariance.
- Bounded Bernoulli/Beta mixtures, including rare-event guardrail coordinates.
- Correlation extremes: strongly positive, negative, near-singular and exactly rank-deficient.
- Heterogeneous coordinate ranges and mixed scale.
- Small sample sizes, where pilot cost can dominate.
- Predictable martingale sequences with fixed conditional mean and valid known sub-Gaussian proxy.
- Mean-shift / changing conditional-mean negative controls: these must be rejected as outside the theorem model, not counted as coverage failures of a model that does not apply.
- Policy-arm imbalance and family-size growth.
- Worst-local-stratum and multi-model rosters.

### Primary metrics

- Simultaneous vector coverage, with exact binomial confidence intervals across independent replications.
- Directional interval coverage for all registered guardrail normals.
- Mean and p50/p95/p99 support-function width by guardrail direction.
- Ellipsoid log-volume and condition number (diagnostics; volume alone is insufficient).
- Width ratios to scalar Bonferroni and the tuned normal-mixture baseline.
- False-safe and false-block rates on known-answer policy mixtures.
- Utility regret and guardrail feasibility after AR-5B26 optimization.
- Pilot/scoring sample cost and fallback frequency.
- Numerical stability, eigenvalue-floor sensitivity, condition number, CPU, memory and deterministic report reproducibility.
- For an anytime method, time-uniform coverage across every inspected time; do not compare a fixed-time confidence set with an anytime confidence sequence as though they were the same target.

### Internal qualification targets (proposed, not achieved or industry standards)

- Simultaneous coverage at or above \(1-\delta\) on the frozen lockbox; publish lower binomial confidence limits, not only observed proportions.
- Zero false-safe selections in the known-answer lockbox.
- Zero positive mixture mass on a policy whose registered metric vector is not estimable.
- On well-conditioned IID vectors, median support-width improvement of at least 20% over scalar Bonferroni at matched simultaneous coverage, without worsening p95 width or false-safe rate.
- Explicit fallback/vacuity for rank-deficient or failed-audit regimes; no silent use of an invalid ellipsoid.
- Correct rejection of changing conditional means and unsupported temporal/cluster regimes.
- Deterministic replay of the same confidence report and support-function bounds.

The 20% target is a research gate, not a claim about current performance. If the method cannot meet it at matched coverage and end-to-end decision quality, keep the simpler scalar method.

## 5. Industry comparison and interpretation

Large experimentation platforms already combine experiment analysis, metric guardrails, staged rollouts and rollback. Spotify's 2026 public materials describe its learning framework and guardrail-based rollback practice. The reported 64% learning / 12% win proportions and 42% guardrail-triggered rollback figure are vendor-reported, use organization-specific definitions and are not directly comparable to our benchmark.

The meaningful comparison is capability and matched task performance:

- Industry systems: experiment lifecycle, metric instrumentation, rollout cohorts, monitoring and rollback.
- This research: confidence-set geometry for correlated decision metrics, robust policy-mixture optimization under confidence bounds, and a verifier that checks the selected mixture.
- Required shared evaluation: false-safe/false-block rate, guardrail regret, sample cost, decision latency and deployment rollback outcomes on the same experiment workload.

Do not claim superiority over Spotify, LaunchDarkly, Statsig, Optimizely or other platforms from paper simulations or a local synthetic fixture. Publish the benchmark harness and invite external reproduction.

## 6. Implementation sequencing

1. Produce a theorem-to-code worksheet for both Candidate A and B with exact assumptions, constants, failure events and fallback paths.
2. Implement the vector mean confidence-set prototype in a separate research module, not inside the current scalar DFCE path.
3. Independently verify support-function bounds and eigenvalue/audit outcomes.
4. Integrate per-policy simultaneous vector events with AR-5B26's hard LP constraints; union-bound all evidence failure probabilities before conditional-alpha spending.
5. Compare at matched coverage to scalar Hoeffding and optimized normal mixtures.
6. Keep fixed-time and anytime APIs as separate types and reports.
7. Add temporal dependence only through a theorem that exactly matches the data-generating process.

## 7. Source references

- Martinez-Taboada & Ramdas, *Empirical Bernstein in Smooth Banach Spaces*, Annals of Applied Probability 36(3), 2199–2236 (2026), DOI: https://doi.org/10.1214/25-AAP2277
- Ramdas & Martinez-Taboada, *Variance-adaptive Chernoff bounds: nonasymptotically valid, asymptotically sharp confidence sets for unbounded means*, arXiv:2610.42058v1: https://arxiv.org/abs/2610.42058
- HiGHS numerical/KKT guidance for future solver-side comparison: https://ergo-code.github.io/HiGHS/dev/guide/numerics/ and https://ergo-code.github.io/HiGHS/dev/guide/kkt/

This document is a benchmark protocol. It does not assert that either research theorem has been independently reimplemented, validated, or shown to outperform the scalar baseline in Symthaea.
