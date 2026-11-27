# CMAQ-033 — ONS-adaptive predictive-rank monitoring

CMAQ-031 provides a fixed-bet predictive-rank martingale portfolio. This crate adds an ONS-adaptive betting kernel so we can compare the fixed-bet baseline with the 2026 predictive-rank-martingale literature.

## Algorithm

For a fixed calibration reference of \(n\) independent cluster-block scores, let \(N_{t-1,j}\) count prior deployment ranks in bucket \(j\). Under the exchangeability null, the next rank obeys

\[
\pi_{t,j} = \frac{1 + N_{t-1,j}}{n+t}.
\]

For a pre-registered feature \(h\), the predictively centered payoff is

\[
Z_t = h((R_t-1)/n) - \sum_{j=1}^{n+1}\pi_{t,j}h((j-1)/n).
\]

Each feature maintains a predictable coefficient \(\lambda_t\) and wealth factor \(1+\lambda_t Z_t\). The maximum possible centered payoff under the current predictive law is

\[
B_t = \max_j |h_j-\sum_\ell \pi_{t,\ell}h_\ell|,
\quad
b_t=\min\{1, 3/(4 B_t)\}
\]

with \(b_t=1\) when \(B_t=0\). After observing the current payoff, update

\[
g_t=\frac{Z_t}{1+\lambda_t Z_t},\quad
A_t=A_{t-1}+g_t^2,\quad
\lambda_{t+1}=\Pi_{[-b_{t+1},b_{t+1}]}
\left(\lambda_t+\frac{(9/2)g_t}{1+A_t}\right).
\]

The bet for observation \(t\) uses only prior ranks; ONS updates the next bet after the current rank is observed. The bound ensures nonnegative factors with a margin above zero. The monitor combines per-feature e-processes using a pre-registered equal-weight mixture and quarantines once the mixture crosses \(1/\alpha\).

For signed ONS betting, one order feature can detect both upward and downward departures, and one dispersion feature can detect expansion and contraction around the reference center. Duplicating each feature with its negative is unnecessary: the sign of the bet already adapts.

## Implementation boundary

OnsPredictiveRankMonitor is an experimental rank-betting kernel. It consumes one-based ranks that must already have been computed by an upstream, evidence-verified comparison against a fixed calibration reference. It does not authenticate the calibration manifest, raw score receipts, tie-break randomness, cluster independence or source authority. Use CMAQ-031 / CMAQ-032 for those lifecycle contracts.

The bound is computed from integer feature numerators and predictive rank counts; final betting wealth and ONS coefficients use IEEE-754 f64. The implementation applies a small downward rounding guard to the projection bound. This is not a formal floating-point proof. The full benchmark must compare conservative numerical implementations and should eventually use a verified interval/fixed-point kernel if this monitor gates physical commitment.

## SOTA comparison and benchmark protocol

The direct prior-art comparator is Kuang & Xia (2026), [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536). They derive the exact fixed-reference predictive rank law, use order and dispersion features, adapt bets with ONS, and report stronger detection than conditional conformal test martingales (CCTM) across their studied synthetic shifts. CMAQ-033 reproduces the core ONS rule for comparison; it does not claim to invent PRMs or ONS.

The frozen protocol should compare:

1. CMAQ-031 fixed-bet predictive-rank mixture;
2. CMAQ-033 ONS Order + Dispersion PRM;
3. ONS Order-only and Dispersion-only ablations;
4. Conditional conformal test martingale (CCTM);
5. Standard conformal test martingale (CTM);
6. Rolling-window breach-rate thresholds (descriptive baseline);
7. Anytime confidence-sequence monitoring of running/rolling risk, where the estimand matches.

### Null experiments

- At least 10,000 independent null replications.
- Alpha in {0.01, 0.05, 0.10}; stream horizons T in {100, 500, 2,000, 10,000}.
- Calibration cluster counts n in {19, 49, 99, 199, 499, 999}.
- Verify empirical probability of ever quarantining and report exact binomial confidence intervals.
- Exercise discrete score ties with independently randomized tie keys; separately test the continuous-score case.
- Include optional stopping and no-lookahead tests.

### Alternative / power experiments

Use pre-registered location, scale/dispersion, center-concentration, heavy-tail, multimodal, gradual, delayed, transient and compound shifts. Also include alternatives orthogonal to the Order and Dispersion features; these must expose, not hide, feature blind spots.

Report:
- detection probability by fixed horizon;
- median, 90th and 95th percentile detection delay conditional on detection;
- average run length under null;
- calibration size vs power curves;
- false-alarm confidence intervals;
- feature-specific and portfolio results;
- cost per update, allocations, memory, p50/p95/p99 latency.

### Prospective targets (not achieved results)

- Preserve theoretical anytime type-I error at the pre-registered alpha under the stated exchangeability null.
- Empirical false-alarm rate no higher than alpha, with a confidence interval that does not conceal underpowered experiments.
- At equal alpha, calibration size and feature set, target at least 20% lower median detection delay than the fixed-bet CMAQ-031 baseline on pre-registered location and dispersion alternatives, without meaningful loss on gradual/delayed shifts.
- For the 2026 PRM paper, first reproduce its reported synthetic protocol and CCTM comparator before making any claim of improvement.
- Keep p99 update latency below 1% of the target planner control-cycle budget on pinned deployment hardware.
- No safety or physical efficacy claim until the full score-ingestion, rank, monitoring, fallback and recovery path has been qualified.

## Statistical caveat

The PRM null is the exchangeability of independent calibration and deployment cluster-block scores. An alarm controls false-alarm probability under that null; a non-alarm does not prove the absence of shift. Power at a fixed calibration size has a fundamental ceiling for arbitrary alternatives. If the score stream is temporally dependent or the deployment shift violates the sampling assumptions, do not report the distribution-free guarantee without an appropriate extension.
