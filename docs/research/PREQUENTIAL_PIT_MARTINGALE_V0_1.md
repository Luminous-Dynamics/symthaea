# Prequential PIT Martingale for Adaptive Experiment Design v0.1

**Status:** research prototype; exact-head Rust validation pending. This module is not claimed as novel statistical theory or superior to established methods. Its purpose is to encode a precise, anytime-valid model-checking contract for Symthaea's adaptive scientific experiments.

## 1. Why a separate detector is needed

The predictive-rank martingale in PR #8452 is distribution-free for a jointly exchangeable calibration/online score sequence, with a marginal guarantee over the reference sample and online stream. An EIG experiment policy, however, changes which physical inputs are queried as observations arrive. Even with a correct physical law, raw residual scores can then have different distributions across query contexts. Feeding those scores into a rank martingale without a justified transformation can invalidate its null.

For a predictive model that can issue a conditional predictive CDF before each outcome, use a prequential probability-integral transform (PIT). For the chosen action (a_t) and past information (mathcal F_{t-1}), let (F_t(\cdot\mid a_t,\mathcal F_{t-1})) be the predictive distribution frozen before observing (Y_t). Define (U_t=F_t(Y_t\mid a_t,\mathcal F_{t-1})) for a continuous predictive law. For discrete outcomes, use the randomized PIT

[
U_t=F_t(Y_t^-\mid a_t,\mathcal F_{t-1})
+W_t\,\Pr(Y_t\mid a_t,\mathcal F_{t-1}),
\]

where (W_t\sim\mathrm{Uniform}(0,1)) is independently randomized as required by the discrete PIT protocol.

Under the null that the conditional predictive distribution is correct, (U_t) is Uniform(0,1) conditional on the past and the selected action. Since the action is chosen before the outcome, it may adapt to earlier observations while preserving that conditional statement. This is a model-based contract: it is only as credible as the predictive CDF, the handling of parameter uncertainty, and the timing/provenance protocol.

## 2. E-process construction

Choose fixed features

[
h_{\mathrm{ord}}(u)=u-\tfrac12,\qquad
h_{\mathrm{disp}}(u)=|2u-1|-\tfrac12.
]

Both have zero mean under (U\sim\mathrm{Uniform}(0,1)) and absolute value at most (1/2). To test conditional failures, predeclare bounded context functions (g_k(a_t,\mathcal F_{t-1})\in[-1,1]), fixed before the current outcome. The module constructs components with payoffs

[
Z_{t,k,f}=g_k(a_t,\mathcal F_{t-1})\,h_f(U_t).
]

The unconditional intercept uses (g_0\equiv 1). Context features are useful where pooled PIT behavior hides context-specific bias. For example, a context that consistently produces PIT 0.75 and another that consistently produces PIT 0.25 can cancel in pooled order statistics; multiplying the order feature by a predictable context sign exposes both deviations in the same direction. Their symmetric PIT values also yield zero dispersion payoff for these exact points.

For each component, the implementation maintains an ONS betting coefficient (\lambda_t\in[-1,1]) and wealth multiplier

[
E_{t,k,f}=1+\lambda_{t,k,f}Z_{t,k,f}.
]

Because ( |Z_{t,k,f}|\leq 1/2 ), each multiplier is at least (1/2). Under conditional PIT validity, (\mathbb E[Z_{t,k,f}\mid\mathcal F_{t-1},a_t]=0), and a predictable (\lambda_t) makes the component a nonnegative martingale. The fixed, equal-weight portfolio over all Order/Dispersion and context channels remains a martingale; its threshold (1/\alpha) therefore yields an anytime false-alarm probability at most (\alpha) under the conditional predictive-model null.

The number of channels and equal portfolio weights are fixed at construction. The module uses stable log-wealth and log-sum-exp calculations and stops permanently after the portfolio crosses the threshold. More channels increase coverage but dilute power: a component with weight (1/m) pays a maximum log-wealth mixture penalty of (log m) relative to that component alone.

## 3. Implementation contract

`prequential_pit_martingale.rs` exposes:

- a fixed alpha and fixed number of context channels;
- validation of PIT range, context vector length, and bounded finite context features;
- an implicit intercept plus Order/Dispersion components for each predeclared context channel;
- ONS updates from prior payoffs, with coefficients clipped to [-1,1];
- per-component log wealth, portfolio log wealth, threshold, observation index, and typed terminal decision;
- tests for invalid inputs, no state consumption on invalid observations, contextual bias that pooled features can hide, and a fixed-seed null smoke simulation with predictable time-varying contexts.

The API cannot verify that a predictive distribution was produced before the outcome, that its parameters were fit without leakage, or that a context feature was computed before seeing (U_t). Record those facts in run receipts. A plug-in CDF that ignores parameter-estimation uncertainty can make PITs non-uniform under the nominal null; it must not be treated as automatically valid. If the forecast is estimated, use a justified prequential/posterior-predictive or honest sample-splitting protocol.

Discrete outcomes require a randomized PIT built from the predictive CDF's left limit and point mass. Passing a raw residual, rank, confidence score, or uncalibrated probability in place of a PIT does not inherit this guarantee.

## 4. How this relates to PRM and other methods

- **PRM (#8452):** distribution-free under a jointly exchangeable reference/online scalar-score stream, with marginal type-I control over that joint sample; not generally conditional on the realized reference sample.
- **Prequential PIT martingale (this branch):** model-based under a *correct conditional predictive distribution*, but can accommodate predictably selected experimental actions via conditional PITs and context multipliers.
- **Finite-horizon Bonferroni monitor (#8450):** simpler batch-rank audit with a fixed horizon; appropriate when repeated marginally valid p-values are available but a full e-process has not been reproduced.
- **Batch conformal audits (#8439):** useful for independent held-out audit batches, subject to exchangeability and calibration resolution.

Classical PIT diagnostics are established in probabilistic-forecast evaluation: Gneiting, Balabdaoui & Raftery, [Probabilistic Forecasts, Calibration and Sharpness](https://doi.org/10.1111/j.1467-9868.2007.00587.x). Adaptive experiment allocation changes inferential assumptions and requires a design-aware protocol; see Bibaut & Kallus, [Demystifying Inference After Adaptive Experiments](https://doi.org/10.1146/annurev-statistics-040522-015431). Modern anytime-valid testing gives the optional-stopping framework, but does not remove the need to state the null correctly; see [Anytime Validity Is Free: Inducing Sequential Tests](https://doi.org/10.1093/jrsssb/qkag050). PRM's fixed-reference alternative is described by Kuang & Xia, [Anytime-Valid Distribution Shift Detection via Predictive Rank Martingales](https://arxiv.org/abs/2609.00536).

These methods are complementary, not directly interchangeable. PRM's distribution-free claim must not be borrowed for PIT tests, and the PIT method's conditional claim must not be stated if predictive distributions are only rough point-estimate fits.

## 5. Benchmark plan

Keep protocol development and confirmatory evaluation separate.

### A. Mathematical/property checks

- Enumerate bounded PIT values and context signs to confirm (|Z|\leq 1/2) and positive multipliers.
- Simulate conditional-Uniform PIT streams under deterministic and past-dependent (predictable) context schedules; estimate the anytime false-alarm rate at alpha = 0.05.
- Include adversarial counterexamples where the current PIT leaks into its own context value; the production contract should reject/flag such provenance rather than claim validity.
- Test discrete outcomes with randomized PITs, score endpoints, numerical boundaries, and terminal alarms.

### B. Conditional-model misspecification

Compare at matched observation/query budgets: prequential PIT Order/Dispersion portfolio; intercept-only PIT portfolio; PRM where the exchangeability assumptions actually hold; batch conformal audit; finite-horizon spending; and simple predeclared residual CUSUM/Page-Hinkley baselines. If reproducing CCTM, keep its stated guarantee and assumptions separate from the others.

Include correctly specified heteroscedastic models under adaptive query choice; mean bias; variance over/under-dispersion; tail misspecification; missing interactions; regime changes; context-specific bias that cancels marginally; parameter-estimation error; and model-refit leakage. Split by held-out structural-world family, not only random seed.

Proposed target: 60+ held-out worlds across at least 6 model/context families and 20 paired seeds per world, plus 10,000 clean-null streams for Monte Carlo error-rate estimates. Report false-alarm estimates with confidence intervals, detection probability/delay curves, measurement/query cost to alarm, post-alarm recovery quality, false-confident-singleton reduction, unresolved rate, and compute overhead. Do not tune features or thresholds on the confirmatory split.

### C. Advancement gates (targets, not results)

- Under the exact conditional-predictive null, empirical false-alarm rate consistent with alpha=0.05 plus the conditional-martingale argument.
- At least 80% detection for preregistered practical conditional-misspecification alternatives at matched cost.
- Detect context-specific misspecification that is intentionally constructed to cancel in pooled PIT diagnostics.
- No more than a 5-percentage-point clean-world recovery loss versus the same discovery policy without the monitor.
- Report cases where the conditional predictive CDF is unknown or poorly estimated as *outside the guarantee*, rather than silently falling back to the theorem.
- Lower median cost to valid recovery or explicit unresolved status than the best simple baseline, with paired world-clustered uncertainty intervals before claiming an improvement.

## 6. Provenance

Every run receipt should include source commit; model and predictive-CDF digest; parameter/uncertainty protocol; experiment/action ID chosen before the observation; predicted CDF version; PIT and any randomized-PIT seed; context feature names and values computed pre-outcome; alpha; component weights; per-feature log wealth; terminal state; raw observation digest; and an evaluator-only artifact for hidden truth. Keep evaluator-only truth and discrepancy labels outside the policy-visible monitor input.

This is research code. The local working environment has no Rust toolchain, so formatting, compilation, and Rust tests remain unverified until focused CI passes on the exact commit. A passing simulation smoke test is not a proof of the model-based assumptions.
