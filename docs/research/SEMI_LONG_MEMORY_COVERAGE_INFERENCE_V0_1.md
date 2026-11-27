# SEMI-LONG-MEMORY-COVERAGE-001 — Gaussian-linear long-memory block sampling v0.1

**Issue:** #14047  
**Implementation:** \`symthaea-semiconductor::benchmark::long_memory_coverage_inference\`  
**Short-memory comparator:** #14046, paper-specific FDM block-means CLT  
**HAC comparator:** #14021, Newey–West/Bartlett  
**Status:** Rust implementation with unit and independent integration tests authored; exact-head validation pending.

## Research source and scope

Zhai, Cheng and Wu, *Conformal Coverage of Time Series: Validity and Inference*, arXiv:2609.33868 (2026-09-27), Section 4 and Appendix C.1–C.4:
https://arxiv.org/abs/2609.33868

The paper proves three different limits for Gaussian linear residuals and gives a paired-block empirical distribution method that estimates the memory parameter and its normalization. This implementation targets that Gaussian-linear model only. It is not a general method for arbitrary long-memory, nonstationary, nonlinear or regime-switching wafer streams.

## Model assumptions

For one fixed predictor, the signed residual sequence must satisfy:

\[
W_t=Y_t-\widehat\mu(X_t)=\sum_{j\ge0}a_j\epsilon_{t-j},
\quad \epsilon_t\stackrel{iid}{\sim}N(0,1),
\]

\[
a_0\ne0,\qquad a_j\sim c j^{-\beta},\quad c>0,\quad \frac12<\beta<1,
\]

with the regular-variation refinement

\[
a_j=j^{-\beta}\{c+O(j^{-\varphi})\},\quad \varphi>0.
\]

The nonconformity score is the exact absolute residual \(V_t=|W_t|\). The predictor must be trained independently of the calibration/test sequence and remain frozen throughout both windows.

The process model evidence record therefore binds:
- nonzero leading coefficient lower bound for \(|a_0|\);
- positive lower bound for the tail coefficient scale \(c\);
- positive refinement exponent \(\varphi\);
- Gaussian iid innovations and regularly varying coefficients declarations;
- frozen predictor artifact and exact absolute-residual score-definition digests;
- source-process-model and assumption-evidence digests.

These are externally declared model assumptions. The finite wafer sequence cannot prove Gaussian innovations, stationarity, regular variation or iid innovations. The evidence digest binds the declaration; it does not authenticate the model issuer or turn an assumption into an observed fact.

## Three distinct limiting regimes

The absolute-value score and threshold-indicator dependence have a phase transition at \(\beta=3/4\):

| Memory parameter | Limit | Coverage error scale |
|---|---|---|
| \(3/4<\beta<1\) | Gaussian | \(n^{-1/2}\) |
| \(\beta=3/4\) | Critical Gaussian | \(\sqrt{\log n/n}\) |
| \(1/2<\beta<3/4\) | Non-Gaussian Rosenblatt-process limit | \(n^{1-2\beta}\) |

The short-memory FDM CLT and HAC normal reference are not valid substitutes in the strong-memory regimes. At \(\beta<3/4\), the limit is non-Gaussian; estimating a standard error alone is insufficient.

## Appendix C.1 implementation

For calibration size \(n\), adjacent test size \(m\), and fixed \(0<\zeta<1\), choose

\[
b=\lfloor n^\zeta\rfloor,\qquad
h=\lfloor\sqrt n\rfloor,\qquad
\ell_b=\lfloor bm/n\rfloor,
\]

\[
k_b=\lceil(b+1)(1-\alpha)\rceil.
\]

Require \(k_b\le b\), \(\ell_b\ge1\), and \(\max(b+\ell_b,2h)\le n\).

For each start \(i=0,\ldots,n-b-\ell_b\), calibrate on the first \(b\) values of that historical block, then evaluate on the immediately following \(\ell_b\) values from the same calibration sequence:

\[
\widehat{\mathrm{Cov}}_{i,b}
=\frac1{\ell_b}\sum_{t=i+b+1}^{i+b+\ell_b}
\mathbf1\{V_t\le\widehat q_{i,b}\}.
\]

These paired subwindows remain entirely inside the historical calibration window; their calibration-to-test ratio matches the target full-window problem.

### Two-scale memory estimate

Use signed residuals, not absolute scores, and overlapping rolling block sums:

\[
\widehat Q_\ell=
\frac1{n-\ell+1}\sum_{i=0}^{n-\ell}
\left(\sum_{t=i+1}^{i+\ell}W_t\right)^2,
\]

\[
\widehat\beta=
\frac32-\frac{\log(\widehat Q_{2h}/\widehat Q_h)}{2\log2}.
\]

If either \(\widehat Q_h\) or \(\widehat Q_{2h}\) is zero, the paper's convention is to set \(\widehat\beta=3/4\). An invalid, non-finite or out-of-range estimate is AssumptionsNotSupported; the implementation never clips it into \((1/2,1)\).

### Normalization

Use the same normalization in all three regimes:

\[
R_N(u)=\left\{\frac{1+\int_1^N x^{2-4u}\,dx}{N}\right\}^{1/2},
\quad \widehat r_N=R_N(\widehat\beta),\quad N=b,n.
\]

The implementation evaluates the integral as

\[
\frac{\exp\{(3-4u)\log N\}-1}{3-4u}
\]

using expm1 for numerical stability, and the continuous \(\log N\) limit at \(u=3/4\).

### Empirical block distribution and inference

The normalized block CDF is

\[
\widehat G_{n,b}(x)=
\frac1{n-b-\ell_b+1}
\sum_{i=0}^{n-b-\ell_b}
\mathbf1\left\{
\frac{\widehat{\mathrm{Cov}}_{i,b}-(1-\alpha)}
{\widehat r_b}\le x
\right\}.
\]

Let \(\widehat c_{\eta/2}\) and \(\widehat c_{1-\eta/2}\) be its empirical quantiles. The full adjacent test-window statistic is

\[
\frac{\widehat{\mathrm{Cov}}_{n,m}-(1-\alpha)}{\widehat r_n}.
\]

Reject the nominal-coverage null only if this statistic lies outside those empirical endpoints. The future realized-coverage reference interval is

\[
\left[
(1-\alpha)+\widehat r_n\widehat c_{\eta/2},
(1-\alpha)+\widehat r_n\widehat c_{1-\eta/2}
\right]\cap[0,1].
\]

No Gaussian critical values or HAC standard errors are used for this long-memory inference.

## Evidence and chronology gate

The implementation reuses the K1-qualified score validation from #14046. Every calibration and test row must:
- be a current K1, numeric-value-bound, revision-attested, full-payload-digest-bound label;
- bind to its immutable prediction receipt and measurement event;
- have label and report availability before delivery and the frozen cutoff;
- have a matching clock domain;
- belong to one contiguous, strictly ordered prediction sequence;
- have one common global model artifact, feature schema and prediction policy;
- have unchanged adapter state within and across both windows.

Missing, UNKNOWN, BLOCKED, stale, unavailable or non-adjacent rows invalidate the complete model sequence; they are not silently filtered to manufacture a shorter stationary sample. The estimand is the declared qualified observed stream—not wafer-population coverage under selective measurement.

## Implementation and tests

Current module contains:
- exact two-scale \(\widehat Q_h,\widehat Q_{2h}\) and unclipped \(\widehat\beta\);
- zero-variance \(\widehat\beta=3/4\) convention;
- expm1-stabilized \(R_N\);
- all three limit regimes;
- paired calibration/test subwindow CDF and empirical quantiles;
- block-constraint checks and minimum paired-block requirement;
- K1/source/clock/sequence/model/policy/adapter-state checks;
- explicit no-IID and no-population-claim flags.

Seven Rust unit tests cover the critical normalization, zero-variance, invalid-beta, regime-boundary, quantile and block-variance contracts. Six integration tests include an independent Appendix C.1 oracle on a seeded synthetic Gaussian-linear power-law residual stream, an explicit out-of-range-beta UNKNOWN control, K1-unqualified and non-adjacent-window rejection, insufficient paired-block handling, and deterministic input-order replay.

## Benchmark tournament

Faithfully reproduce the paper's Study 3:
- \(\beta\in\{0.9,0.75,0.6\}\), plus \(0.7\) as an additional ultra-long-memory case;
- \(n\in\{2^{12},2^{14},2^{16},2^{18},2^{20}\}\), \(m=n\);
- \(h=\lfloor\sqrt n\rfloor\), \(b=\lfloor n^{2/3}\rfloor\);
- independent reference trajectories for finite-sample quantiles;
- no empirical recentering or rescaling of the normalized distribution;
- at least 1,000 Monte Carlo replications where feasible, with confidence intervals for rejection rates.

Compare separately against:
- exact Appendix C block sampling;
- #14046 short-memory FDM block-means normal inference (assumptions violated at \(\beta\le3/4\));
- #14021 HAC/Bartlett;
- moving-block bootstrap;
- IID normal/binomial diagnostics.

Report Type I error, power under 1/2/5 percentage-point undercoverage, prediction-interval coverage, block-CDF/quantile error, \(\widehat\beta\) error, normalization error, runtime and memory.

The paper's simulation reports a major distinction under \(\beta=0.6\): at the largest sample point shown in its rejection-rate table, the block-sampling test is 5.65% at a nominal 5%, compared with 44.35% for moving-block bootstrap and 43.77% for HAC. This is a paper-reported result, not a result from our implementation. Our reproduction must establish whether it can match that result under the exact paper design.

## Claim ceiling

This is a software reproduction of one paper-specific Gaussian-linear long-memory model. It does not establish that a real fab residual stream is Gaussian linear or regularly varying; it does not handle arbitrary nonlinear long memory, changing predictors, structural breaks or informative metrology selection. It does not authenticate process-model assumptions, identify population coverage or authorize equipment/wafer decisions.
