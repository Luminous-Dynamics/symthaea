# Predictive-Rank Martingale Detector v0.1

**Status:** implementation-oriented research prototype; exact-head Rust CI is pending. No reproduced literature result, comparative performance claim, or state-of-the-art claim is made.

This branch adds a separate sequential detector on top of the finite-horizon monitor in PR #8450. It implements the Order/Dispersion predictive-rank martingale core described by Kuang and Xia, *Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales* (preprint dated September 2026): https://arxiv.org/abs/2609.00536.

## 1. Paradigm shift: monitor predictive ranks, not a repeated batch p-value

A standard conformal rank p-value is useful for a single independent audit. Repeating a one-shot 5% test at every time point does not preserve a 5% false-alarm budget. The finite-horizon baseline in PR #8450 handles that with Bonferroni spending but becomes conservative for long horizons and needs a minimum calibration-set size for its threshold to be attainable.

Predictive-rank martingales take a different path: preserve the fixed calibration reference and model the *conditional null distribution of each online rank given earlier ranks*. Betting against that predictive distribution yields a nonnegative martingale under the declared null assumptions. A wealth threshold then supports anytime monitoring without predeclaring a finite horizon.

This implementation uses two complementary rank features:
- **Order:** (h_{ord}(u)=u-0.5), sensitive to persistent low/high ranks such as location shifts.
- **Dispersion:** (h_{disp}(u)=|2u-1|-0.5), sensitive to extreme ranks at either end, including symmetric scale/tail changes that can cancel in a directional statistic.

The portfolio is an equal mixture of both wealth processes. The feature definitions are fixed for the whole epoch; this is not an adaptive search over a catalogue of tests.

## 2. Conditional rank law and e-process

Let the fixed reference have (n) scores, and let (R_t\in\{1,\ldots,n+1\}) be the one-based rank of online score (t) against that reference, using score/tie-breaker lexicographic order. Let (N_{t-1,j}) be the number of earlier online scores assigned rank (j). The paper's conditional predictive rank probabilities are

[
\pi_{t,j}=\frac{1+N_{t-1,j}}{n+t},\qquad j=1,\ldots,n+1.
]

For a predeclared rank feature (h), define

[
Z_t=h\left(\frac{R_t-1}{n}\right)
 - \sum_{j=1}^{n+1}\pi_{t,j}h\left(\frac{j-1}{n}\right).
]

With a predictable betting coefficient \(\lambda_t\) constrained by the paper's feature-specific bound, each multiplier (1+\lambda_t Z_t) is nonnegative and has conditional expectation one under the null. Multiplying the factors therefore yields a nonnegative martingale. The module tracks log wealth and raises an alarm when

[
\log M^{OD}_t \geq -\log(\alpha),
\quad M^{OD}_t=\tfrac12 M^{ord}_t+\tfrac12 M^{disp}_t.
]

The ONS betting update in the paper uses (g_t=Z_t/(1+\lambda_t Z_t)), cumulative (A_t=\sum_{s\leq t}g_s^2), and the next predictable projection bound derived from the updated rank counts. The Rust implementation uses the paper's (9/2) update constant and computes portfolio wealth with stable log-sum-exp arithmetic.

## 3. Implementation contracts

`predictive_rank_martingale.rs` provides a standalone implementation with no additional crate dependency:

- fixed calibration scores and explicit tie-breakers, sorted by lexicographic order;
- validation of alpha, score finiteness, tie-breaker range ([0,1)), lengths, duplicate reference keys, and exact audit/reference key collisions;
- the conditional rank probabilities and rank-count updates;
- the Order and Dispersion conditional payoffs;
- online Newton-step betting updates with predictable clipping bounds;
- separate per-feature log wealth and a numerically stable 50/50 portfolio log wealth;
- a terminal alarm state that cannot be silently resumed using the same epoch;
- focused unit tests for error handling, initial conditional centering, a persistent upper-rank shift, symmetric extreme ranks, and invalid observations not mutating the epoch.

The stated anytime type-I guarantee is **marginal over the jointly sampled calibration reference and online stream**, as in the paper's theorem. It is not generally conditional on a particular realized numeric calibration sample. The conditional filtration used by the theorem is the history of ranks, not arbitrary information from the raw calibration values or online scores. This distinction matters in deployment: the reference must be a representative random sample from the intended clean population, the scalar scoring rule must be frozen before calibration/monitoring, and detector-side adaptation must use only the permitted rank history. If a use case requires a guarantee conditional on a particular fixed reference dataset, this implementation does not establish it.

All callers must use one fixed score function and target/reference protocol per epoch. The caller-supplied tie breaker is validated but **not proven** to be an independent uniform draw. Atomic scores require independent random tie breaking for the rank law to apply. Record the tie-breaker generator and seeds/provenance in the run receipt. Exact key collisions with a calibration item are rejected rather than given an arbitrary tie convention.

A monitoring alarm signals a change in the monitored score/rank process under the assumptions; it does not diagnose the cause, prove a candidate equation false, or choose the next physical model by itself. Integrating the result into discovery policy should use a typed transition to independent investigation/model expansion/validation. A non-alarm is not proof of scientific adequacy.

## 4. Literature comparison and what is not implemented

Kuang & Xia's preprint reports experiments with (n=1000) calibration scores and (α=0.05), including immediate and delayed location shifts, gradual shifts, and symmetric alternatives such as wider Gaussian, Laplace, and standardized Student-t distributions. The authors report that their Order PRM reached 80% detection with 10.2–23.1% fewer observations than CCTM in nine studied location-shift settings, while the Dispersion feature/portfolio detected symmetric alternatives where the directional Order statistic had near-zero power. Those are the paper's results, **not results from Symthaea**.

Relevant comparisons:
- Kuang & Xia, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536): primary algorithm to reproduce.
- Shaer et al., ICML 2026, [Testing For Distribution Shifts with Conditional Conformal Test Martingales](https://proceedings.mlr.press/v306/shaer26a.html): strong comparator with a fixed-reference conditional test-martingale design.
- Lee, UAI 2026, [Diagnosing Conformal Prediction Failures Under Distribution Shift](https://proceedings.mlr.press/v337/lee26g.html): motivates predictor-specific shift analysis rather than a one-size-fits-all residual claim.
- Callahan & Catanach, AISTATS 2026, [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html): explains why information-maximizing experimental design can amplify inferential bias under model misspecification.
- Tang, Sloman & Kaski, AISTATS 2026, [Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html): adjacent robust acquisition comparator, not implemented by this detector.

This module is a reproduction-oriented implementation of the published PRM core, not an exact reproduction of every experiment or all baselines in the paper. CCTM itself is not implemented here. No claim is made that the combined Symthaea acquisition/audit/monitoring pipeline is better than these methods.

## 5. Confirmatory benchmark protocol

Build two tiers and freeze the protocol before inspecting held-out results.

### Tier A — reproduce the method-level paper settings

Use (n=1000) calibration scores, (α=0.05), and 1,000 independent repetitions per setting, matching the reported scale of the preprint's evaluation. Recreate:

- immediate Gaussian mean shifts at standardized shift sizes (d\in\{1,1.5,2\});
- delayed shifts with change times (t_0\in\{200,600,4000\}) and post-change mean 2;
- gradual shifts with the paper's reported rates (verify exact generator details from the paper before coding);
- symmetric alternatives including (N(0,1.5^2)), (Laplace(0,1/\sqrt{2})), and standardized (t_3);
- the same detector thresholds and stopping conventions.

Compare PRM Order, PRM Dispersion, PRM Order/Dispersion portfolio, and an independently reproduced CCTM baseline. Match the calibration data and the online streams across methods wherever the published protocol permits. Report full detection-probability curves over time, average run length under the null, false-alarm probability, median/90th-percentile detection delay, calibration size, wall-clock and memory, and confidence intervals. Verify paper code/license availability before importing an implementation.

### Tier B — scientific-law discovery under model misspecification

Extend beyond generic shift detection to the Symthaea physics discovery setting. Use held-out structural-world families rather than only new seeds of the same generator:

- correctly specified controls and omitted structural terms;
- correlated and heavy-tailed measurement noise;
- heteroscedastic noise, score-distribution shifts, and abrupt/gradual regime changes;
- rare-region and extrapolation errors which can hide in a pooled RMS residual;
- atomic/repeated score values to test tie-breaking and numeric stability;
- invalid predictions and malformed/incomplete provenance receipts.

Suggested minimum: at least 60 held-out worlds across 6 or more structural families and 20 paired seeds per world. Keep policy-visible scores separate from evaluator-only truth and discrepancy labels. Compare at matched measurement budgets against: fixed-horizon Bonferroni (#8450), batch conformal audit (#8439), simple residual CUSUM/Page-Hinkley, PRM Order, PRM Dispersion, the portfolio, and reproduced CCTM. The experiment acquisition comparison should separately include nominal EIG and scenario-robust EIG (#8361); do not confound acquisition quality with detector quality.

### Advancement gates (targets, not results)

- Type-I control consistent with the chosen method's assumptions, verified via at least 10,000 clean-null streams with intervals and a separate analytic/protocol argument.
- Report power versus change magnitude and detection delay, rather than one chosen shift.
- No more than a 5-percentage-point clean-world scientific-recovery decline compared with the same discovery policy without the detector.
- At least 20% relative reduction in false confident singleton outcomes under held-out misspecification without increasing unresolved outcomes indefinitely.
- Lower median total measurement cost to valid recovery or explicit unresolved status than the best simple baseline, using paired world-clustered intervals.
- All negative controls and all failed seeds preserved; no post-hoc threshold or feature selection on the confirmatory split.

Passing these targets would support a task-specific comparative claim, not a universal SOTA claim.

## 6. Receipts and validation status

A run receipt should bind: source commit; paper/protocol version; reference scores and tie-breaker digest; score-function version; target population; alpha; detector/feature configuration; online tie-breaker source; online observation IDs; per-observation rank and log wealth; stopping decision; measurement costs; seeds; and raw artifact digest. Store hidden simulation truth in a separate evaluator-only artifact.

The module is covered by the focused expected-information-gain workflow, including formatting and module tests. This working environment has no Rust toolchain. Until CI completes on the **exact** branch head, formatting, compilation, and Rust tests remain unverified. Queued/pending/skipped checks are not passes.
