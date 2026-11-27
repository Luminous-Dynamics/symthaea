# SEMI-FDM-COVERAGE-001 — paper-specific realized-coverage inference v0.1

**Issue:** #14014  
**Implementation:** `symthaea-semiconductor::benchmark::fdm_coverage_inference`  
**Stacked baseline:** #14021, Newey–West/Bartlett HAC (separate estimator)  
**Status:** Rust implementation with unit and independent integration tests authored; exact-head validation pending.

## Research source

Zhai, Cheng and Wu, “Conformal Coverage of Time Series: Validity and Inference,” arXiv:2609.33868 (27 September 2026): https://arxiv.org/abs/2609.33868

The implementation follows:
- equation (1): exact split-conformal rank (k_n=\lceil(n+1)(1-\alpha)\rceil), with (\hat q_n=+\infty) if (k_n>n);
- equations (2)–(4): adjacent test realized coverage and indicator functional-dependence coefficients (\theta_k), (\Theta=\sum_{k\ge0}\theta_k);
- Theorem 1, equation (5): density-free marginal-coverage error bound;
- Theorem 3, equation (9): realized-coverage CLT with long-run variance of threshold indicators;
- Proposition 4, equation (10) and its stated tail condition: non-overlapping calibration-block variance estimator;
- the implementation does **not** claim the paper's Section 4 long-memory inference or Appendix C block-sampling distribution method.

## Why this is distinct from HAC

The HAC baseline in #14021 estimates a truncated Bartlett sum of lagged covariances from the observed test miss sequence.

This implementation instead:
1. computes the exact conformal cutoff from the calibration scores;
2. evaluates calibration indicators at that cutoff;
3. forms non-overlapping block means from the calibration indicators;
4. estimates the indicator long-run variance using the paper's batch-means formula;
5. scales that estimate by (\sqrt{1/n+1/m}) to include calibration and test uncertainty.

For calibration indicators (I_t=\mathbf 1\{V_t\le\hat q_n\}), block length (\ell), and (B=\lfloor n/\ell\rfloor), it uses exactly:

\[
\bar I_j=\frac{1}{\ell}\sum_{t=(j-1)\ell+1}^{j\ell} I_t,\qquad
\bar I=\frac{1}{B}\sum_{j=1}^{B}\bar I_j,
\]

\[
\hat\sigma_{\rm cov}^{2}
=\frac{\ell}{B}\sum_{j=1}^{B}(\bar I_j-\bar I)^2,
\qquad
\widehat{SE}=\hat\sigma_{\rm cov}\sqrt{\frac1n+\frac1m}.
\]

The divisor is (B), and incomplete trailing calibration blocks are discarded, exactly as in the paper. This is not a Bartlett lag window, Newey–West estimator or moving-block bootstrap.

## Indicator FDM evidence contract

For the score-indicator process the paper defines:

\[
\theta_k=\sup_v\left\|\mathbf1\{V_k\le v\}
-\mathbf1\{V_k^*\le v\}\right\|_2,\qquad
\Theta=\sum_{k=0}^{\infty}\theta_k.
\]

The code requires a process-model evidence record containing:
- the frozen predictor artifact digest and exact score-definition digest;
- the stationary causal innovation representation and iid-innovation assumptions;
- an explicit claim that the predictor was trained independently of calibration and test;
- continuity and quantile-density/no-ties assumptions for the CLT;
- finite upper bounds for (\theta_0,\ldots,\theta_K);
- a tail envelope (\theta_k\le C k^{-a}), (k>K), with (a>3);
- a digest binding the external assumption evidence.

From these inputs the implementation computes conservative upper bounds for (\Theta) and the stronger Proposition 4 tail condition:

\[
\sum_{k\ge1}\left(\sum_{j\ge k}\theta_j\right)^{1/2}<\infty.
\]

The (a>3) power-tail condition is sufficient for this variance-estimation condition. The program does **not** estimate (\theta_k) from observed wafer residuals, prove the innovation representation, prove stationarity, or authenticate the process-model evidence. A digest only binds the declared record.

## Marginal-coverage bound

The code also evaluates the paper's density-free sufficient condition:

\[
\frac2n+\left(\frac{2\Theta^2}{n}\right)^{1/3}
<\min(\alpha,1-\alpha),
\]

and, only when it holds, reports:

\[
\left|\Pr\{Y_{n+1}\in\hat C_n(X_{n+1})\}-(1-\alpha)\right|
\le \frac2n+3\left(\frac{\Theta^2}{4n}\right)^{1/3}.
\]

This is a deterministic-future-index marginal-coverage error bound under the stated FDM assumptions. It is not conditional per-wafer coverage and not a population result under selective metrology.

## Frozen predictor and adjacent windows

The input contains two separately manifested windows:
- calibration score observations;
- immediately adjacent test score observations.

Every score observation binds:
- immutable streaming prediction receipt;
- global model artifact, feature schema, prediction-policy and adapter-state digests;
- K1 report, label ID/revision, full label-payload digest, measurement event and corrected measurement;
- label availability, report availability, delivery, clock domain and cutoff.

The evaluator requires:
- one common model/schema/policy/adapter-state identity;
- no adapter-state change during the calibration/test windows;
- exact contiguous prediction sequence across the calibration-to-test boundary;
- strictly increasing event time;
- every physical label and K1 report available by the frozen cutoff;
- every label is current K1, eligible and downstream-consumable;
- no duplicate samples, events, label revisions or prediction sequences.

Any unknown, blocked, stale, missing, selected-out, late or unavailable observation makes the paper's complete-stream assumption unsupported; it is not silently dropped to manufacture a shorter apparently stationary series. The estimand remains the declared fully observed qualified stream. It does not become wafer-population coverage.

## Long-memory boundary

The paper's Section 4 and Appendix C are a different algorithm:
- Gaussian linear residual process (W_t=\sum_{j\ge0}a_j\epsilon_{t-j}), (a_j\sim c j^{-\beta}), (1/2<\beta<1);
- three regimes split at (\beta=3/4);
- a Gaussian limit for (\beta>3/4), logarithmically normalized Gaussian limit at (\beta=3/4), and a non-Gaussian Rosenblatt-process limit for (1/2<\beta<3/4);
- a two-scale estimate of (\beta) and paired overlapping calibration/test block-sampling distribution in Appendix C.

None of that long-memory inference is implemented here. The report sets `long_memory_supported=false`. In particular, the short-memory CLT, HAC baseline and this Proposition 4 block-SE must not be applied to the strong-memory regimes as if they shared the same limit law.

## Benchmark reproduction matrix

Reproduce the paper's short-memory study first:
- ARCH(1) and stochastic-volatility processes;
- (m=n);
- non-overlapping calibration blocks (\ell=\lfloor n^{2/3}\rfloor);
- exact paper rank and adjacent test window;
- no empirical recentering or rescaling before assessing the studentized distribution.

Then stress:
- IID scores;
- the paper's non-mixing but geometrically FDM-decaying recursion;
- ARCH and stochastic volatility;
- AR(1) residuals and negative/positive dependence;
- chamber-common shocks and common-reference cohorts;
- Student-t5 common-scale shocks;
- regime shifts as **assumption-violation controls**, not covered by the stationary theorem.

Baselines remain distinct:
- IID/Wilson diagnostic (deliberately naive under dependence);
- PR #14021 Newey–West/Bartlett HAC;
- this exact paper-specific block-means short-memory estimator;
- moving/circular/stationary block bootstrap;
- Kupiec, Christoffersen and Dynamic Quantile backtests.

Targets are empirical size near the nominal 5% level, correct realized-coverage interval coverage, power under fixed undercoverage alternatives and honest UNKNOWN on assumption failure. Use paired seeds and independent reference trajectories. Do not calibrate on the same Monte Carlo trajectories used to report rejection rates.

## Claim ceiling

This is a paper-specific short-memory inference implementation with externally declared assumptions. It does not verify the FDM certificate, infer arbitrary dependence from one stream, support long-memory inference, correct physical metrology selection, identify population risk, authenticate physical labels, or authorize manufacturing decisions. The HAC baseline and this estimator are alternative inferential methods with distinct assumptions; neither is a process-control signal.
