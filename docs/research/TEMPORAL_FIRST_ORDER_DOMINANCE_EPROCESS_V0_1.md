# Temporal first-order stochastic dominance e-process v0.1

**Tracking:** NEURO-SILICON-015 / #9930  
**Implementation:** \`crates/domains/symthaea-architecture-engineering/src/temporal_stochastic_dominance.rs\`  
**Composes:** temporal intervention family #9932/#9929; temporal distributional empirical reference #9933/#9930; generic family ledger #9465; paired intervention audit #9881.  
**Status:** mathematical reference implementation and tests authored; no Rust compilation, test, CI, or real-data benchmark execution.

## 1. Estimand and direction

For each paired episode, define reward as negative exposure:

\[
X_t=-Y^{control}_t,\qquad Y_t=-Y^{treatment}_t.
\]

Higher reward is better, so lower exposure is better.

The null tested by this module is:

\[
H_0:\quad Y\preceq_1 X,
\]

meaning the treatment reward is first-order stochastically dominated by the control reward. Intuitively, treatment has no distributional upside over control.

The alternative is that there exists at least one threshold \(z\) with:

\[
F_X(z)-F_Y(z)>0.
\]

Rejecting this null means **evidence of treatment upside at one or more thresholds**. It does **not** prove that treatment first-order dominates control at every threshold. A treatment can have some upside and still have a worse tail elsewhere. The empirical CDF and stop-loss reference in \`distributional.rs\` remains necessary for inspecting the full risk profile.

This is a one-sided stochastic-dominance test, not the paired sharp no-effect test in #9881 and not a weak average treatment-effect test.

## 2. Building-block e-variable

At each predeclared finite-support threshold \(z\), define:

\[
D_t(z)=\mathbf 1\{X_t\le z\}-\mathbf 1\{Y_t\le z\}\in\{-1,0,1\}.
\]

Under \(H_0\), \(E[D_t(z)]\le0\). For any predictable \(\lambda_t(z)\in[0,1]\),

\[
S_t(\lambda_t(z),z)=1+\lambda_t(z)D_t(z)
\]

is a nonnegative e-variable under the paper's IID-pair sampling assumptions.

For a finite known exposure support, the module uses every non-maximum reward threshold, which is the complete finite threshold set where a CDF violation can occur. It assigns a fixed uniform threshold mixture:

\[
S_t=\frac{1}{m-1}\sum_{j=1}^{m-1}
\left(1+\widehat\lambda_t(z_j)D_t(z_j)\right).
\]

The running product \(E_t=\prod_{s=1}^{t}S_s\) is an e-process. The rejection boundary is \(1/\alpha_{stream}\), where stream alpha is derived from the generic preregistered family manifest as family alpha divided by the full contrast × three-axis stream count.

## 3. Predictable growth-rate-optimal plug-in

For each threshold, let \(p_t(z)\) and \(q_t(z)\) be the prior counts of positive and negative paired crossings:

- positive: \(X_s\le z<Y_s\);
- negative: \(Y_s\le z<X_s\).

Before observing pair \(t\), the implementation computes:

\[
\widehat\lambda_t(z)
=
\min\left(1-c,\,
\left(\frac{\widehat p_{t-1}(z)-\widehat q_{t-1}(z)}
{\widehat p_{t-1}(z)+\widehat q_{t-1}(z)}\right)_+
\right),
\]

with zero when the denominator is zero. \(c>0\) is a predeclared clipping constant. The current pair is scored only after these bets are fixed, so the update is predictable.

This follows the finite-threshold GRO plug-in construction in Arnold, Choe, Scarsini and Tsetlin (2026), Section 3.3. In their Theorem 2, power-one behavior requires the predictable threshold mixture to retain mass bounded away from zero on the violation region. For finite known support, a fixed uniform mixture has full support and satisfies this requirement for any nonempty violation set. This implementation therefore uses the fixed uniform mixture rather than making the paper's adaptive EWA weighting its default.

The paper notes that, on finite known support, its adaptive threshold weighting can improve on uniform weighting by at most an \(O(\log m)\) factor. That makes uniform weighting an appropriate transparent baseline before experimenting with more adaptive mixtures.

## 4. Preregistered finite support and family identity

Each temporal contrast/axis has one strictly increasing finite exposure support in microseconds. Every support profile is bound to the exact generic stream ID, and a domain-separated SHA-256 digest commits to the support values.

The complete stochastic-dominance family wraps the generic temporal confidence family and commits to all support profiles. The generic manifest owns the family alpha and number of contrast-axis streams; callers cannot pass a smaller family-size integer.

The profile must be externally committed before outcomes are observed. The hash does not establish wall-clock chronology or signer custody.

Pairs are rejected unless:
- they carry nonzero pair/context/source-family/assignment-receipt identities;
- pair, context and assignment-receipt IDs are unique in the stream;
- assignment is declared independent fair-coin at 500000 PPM;
- the episode horizon is positive and both arm exposures lie within it;
- both arm exposures are members of the frozen finite support;
- the pair arrives at the exact next sequential look.

The e-process assumes paired observations are IID across episodes; arbitrary dependence within a pair is allowed. Fair-coin randomization alone does not make an arbitrarily drifting or dependent time sequence IID. Do not mix incompatible regimes into one stream.

## 5. Numerical implementation boundary

The mathematical e-process is defined over exact real-valued increments. The Rust prototype accumulates log wealth in f64 and subtracts a conservative engineering guard before checking the boundary. The guard is **not** a formal correctly-rounded arithmetic proof. A high-assurance verifier should recompute the path with higher precision or directed interval arithmetic before using a crossing as an authoritative receipt.

Therefore the current implementation is a research reference, not a production certification engine. It also does not persist an independent immutable receipt chain; the runner must bind the pair sequence and report to the existing receipt authority before any decision is promoted.

## 6. Tests authored

Twelve Rust tests currently cover:
1. complete support-profile coverage and digest mutation;
2. missing support-profile rejection;
3. derived family alpha and threshold boundary;
4. first bet uses only prior data;
5. persistent treatment lower-exposure advantage eventually crosses the no-upside boundary;
6. deterministic equal-exposure null produces no evidence;
7. blocked assignment, exposure outside support and duplicate pairs fail closed;
8. sequential look gaps do not advance evidence;
9. one-step supermartingale expectation on a non-degenerate IID stochastic-dominance null across all short histories;
10. non-fair assignment probability rejection;
11. deterministic reconstruction from the generic temporal pair ledger;
12. rejection when a pair exposure binding is tampered.

These are authored tests, not executed tests.

## 7. Benchmark and research comparisons

Compare on identical paired episodes, frozen support profiles and the same predeclared family:
- fixed uniform threshold mixture with predictable GRO plug-in (this implementation);
- constant \(\lambda=0.1\) uniform threshold mixture;
- fixed uniform threshold mixture with a small predeclared betting-fraction portfolio;
- paper's AdaGRO-Exp / AdaGRO-Linear / AdaGRO-Greedy after the theorem's power-one condition is checked for the actual weighting sequence;
- fixed-sample paired permutation / KS-style tests at a predeclared terminal sample;
- naive repeated fixed-sample tests as an invalid optional-stopping negative control.

Report false-crossing probability, power, time-to-crossing, inconclusive rate, mean/median, empirical CDF, p90/p95/p99, stop-loss curve, source-family/regime strata and resource cost. The sequential test answers only the declared no-upside null; it does not replace full empirical distribution reporting.

## 8. Claim ceiling

A software PASS would establish implementation semantics for the specified finite-support paired process and its test cases. It would not establish population stochastic dominance outside the IID paired sampling assumptions, absence of unmeasured confounding, causal generalization, tail safety, nonstationary-stream validity, hardware efficiency or SOTA superiority.

Reference:
Arnold, Choe, Scarsini & Tsetlin (2026), *Betting on Bets: Anytime-Valid Tests for Stochastic Dominance*, arXiv:2604.21851, especially Theorem 1, Proposition 1, Theorem 2 and Remark 5:
https://arxiv.org/abs/2604.21851
