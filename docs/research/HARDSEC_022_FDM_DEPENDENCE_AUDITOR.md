# HARDSEC-022 — Functional-Dependence Qualification and Block Coverage Inference

**Status:** Rust FDM-contract and block-diagnostic auditor authored with synthetic fixtures; model assumptions remain unverified and no physical PMU dependence proof is claimed  
**Issue:** [#9605](https://github.com/Luminous-Dynamics/symthaea/issues/9605)  
**Integrates conceptually with:** [HARDSEC-019 DKW-CCTM](https://github.com/Luminous-Dynamics/symthaea/pull/9478), [HARDSEC-020 context assignment](https://github.com/Luminous-Dynamics/symthaea/pull/9484), [HARDSEC-021 restarted CCTM](https://github.com/Luminous-Dynamics/symthaea/issues/9593)

## 1. Why another dependence layer is needed

HARDSEC-019/021 correctly refuse to infer independence or effective sample size from window counts, caller-declared group IDs or a digest. Hardware performance-counter streams can be strongly dependent across adjacent windows, especially with overlapping windows, cache/branch predictor state, scheduling, frequency changes, and repeated execution phases.

Raw autocorrelation is not a proof of the dependence property required by a sequential conformal test. In particular, the dependence of raw counters, anomaly scores, and threshold indicators can differ. A detector can have weak-looking score autocorrelation while its threshold-indicator process remains strongly dependent.

The target paradigm is therefore **innovation influence accounting**: declare a causal model, describe how perturbing one innovation can change future score-threshold indicators, bound that influence across lags, and keep the assumptions/source proof explicit. Where that model-bound route is unavailable, report block-based diagnostics on realized calibration coverage without promoting them to anytime-valid evidence.

## 2. Research basis

Zhai, Cheng and Wu (2026), [*Conformal Coverage of Time Series: Validity and Inference*](https://arxiv.org/abs/2609.33868), define a functional-dependence measure (FDM) by replacing one innovation in a causal time-series representation with an independent copy and measuring the change in future score-threshold indicators. They derive non-asymptotic marginal coverage bounds without relying on mixing assumptions, then develop realized-coverage inference and a consistent non-overlapping block variance estimator under additional dependence and regularity conditions. They also show that long-memory settings can have non-Gaussian coverage limits.

The score-threshold indicator FDM is:

[
\theta_k=\sup_v\left\|\mathbf 1\{V_k\leq v\}-\mathbf 1\{V_k^*\leq v\}\right\|_2,
\qquad
\Theta=\sum_{k=0}^{\infty}\theta_k.
]

Here (V_k^*) is the score after replacing one past innovation with an independent copy. This is a model-level coupling quantity, not an ordinary empirical ACF.

Under the paper's assumptions, its Theorem 1 gives the following density-free marginal coverage bound when the stated numeric condition holds:

[
\frac{2}{n}+\left(\frac{2\Theta^2}{n}\right)^{1/3}
<\min(\alpha,1-\alpha)
]

and then:

[
\left|\Pr\{Y_{n+1}\in\widehat C_n(X_{n+1})\}-(1-\alpha)\right|
\leq
\frac{2}{n}+3\left(\frac{\Theta^2}{4n}\right)^{1/3}.
]

This is a **marginal coverage error bound for the fixed score-threshold procedure** under its model assumptions. It is not a sequential CCTM PFA bound.

For realized coverage over adjacent calibration/evaluation windows, the paper's block estimator uses (B=\lfloor n/\ell\rfloor) complete non-overlapping calibration blocks:

[
\overline I_j=\frac1\ell\sum_{t=(j-1)\ell+1}^{j\ell}
\mathbf 1\{V_t\leq\widehat q_n\},
\qquad
\widehat\sigma^2_{\mathrm{cov}}
=\frac{\ell}{B}\sum_{j=1}^{B}(\overline I_j-\overline I)^2.
]

The asymptotic standard error is:

[
\widehat{\mathrm{SE}}
=\widehat\sigma_{\mathrm{cov}}\sqrt{\frac1n+\frac1m}.
]

We also report the iid reference standard error (\sqrt{\alpha(1-\alpha)(1/n+1/m)}) for comparison. It is not used as a substitute for the block estimate.

Barber and Pananjady (2026), [*Predictive inference for time series: why is split conformal effective despite temporal dependence?*](https://proceedings.mlr.press/v313/barber26a.html), provide a complementary switch-coefficient analysis of split-conformal coverage loss under stationary beta-mixing processes. This is another comparator for later work; it does not give a general sequential e-process guarantee.

## 3. FDM contract auditor

Rust executable: `examples/hardsec_temporal_dependence_audit.rs`  
Harness feature: `hardsec-temporal-dependence-audit`

Inputs:
- Model/assumption manifest: `schemas/hardsec-temporal-dependence-model-v1.schema.json`
- Calibration/evaluation score stream: `schemas/hardsec-temporal-dependence-data-v1.schema.json`
- Fixture model: `tests/fixtures/hardsec/temporal-dependence-v1/model.json`
- Fixture scores: `tests/fixtures/hardsec/temporal-dependence-v1/data.json`

Version 1 accepts a claimed geometric upper envelope:

[
\theta_k\leq A\rho^k,\quad A>0,\quad0<\rho<1.
]

For declared per-lag upper bounds through lag (L), it checks:
- lags are contiguous from zero;
- each bound lies in ([0,1]);
- each supplied bound is below the declared geometric envelope;
- the proof artifact has a valid SHA-256 syntax;
- causal model, innovation iid, stationarity, fixed-predictor and FDM proof statuses are all left as `assumption_not_verified`.

It computes the finite partial sum plus analytic tail:

[
\Theta_{\mathrm{upper}}
=\sum_{k=0}^{L}\theta_{k,\mathrm{upper}}
+\frac{A\rho^{L+1}}{1-\rho}.
]

A `not_bounded` manifest is accepted as a valid descriptive input, but the receipt emits no FDM sum or theorem bound. A geometrically consistent manifest receives `model_bound_summable_fdm_contract_consistent_not_authenticated`, not a verified theorem status. The auditor checks arithmetic and input consistency; it does not prove the causal representation or authenticate the model/proof artifact.

## 4. Block-based realized-coverage diagnostic

The tool:
1. computes the fixed upper-order-statistic cutoff (\widehat q_n=V_{(\lceil(n+1)(1-\alpha)\rceil)});
2. builds calibration and evaluation indicators (I_t=\mathbf1\{V_t\leq\widehat q_n\});
3. uses each preregistered block length with complete blocks only;
4. reports block count, discarded tail count, block means, the covariance-sum estimate, block standard error, iid standard error, ratio, realized coverage and standardized deviation;
5. reports a normal-reference prediction band centered on the nominal coverage, not on the observed coverage;
6. reports sensitivity across block lengths.

The normal interval is marked `posthoc_asymptotic_diagnostic_only_assumptions_unverified`; the small-block Student-t diagnostic is not implemented in v1. A block standard error or confidence interval is not an e-value, a martingale, or an anytime-valid false-alarm guarantee.

## 5. Synthetic known answer

The fixture uses 2,000 calibration scores and 2,000 evaluation scores, target miscoverage (\alpha=0.10), three block lengths (25, 50, 100), and distinct declared calibration/evaluation execution and device-boot IDs.

Declared geometric envelope:
- (A=0.05);
- (\rho=0.8);
- (\Theta_{\mathrm{upper}}=0.25).

The numerical condition in Theorem 1 is (0.0406850<0.10), and the resulting marginal coverage error bound is approximately 0.0605275 **if the causal model and FDM assumptions hold**. Those assumptions are explicitly unverified.

The calibration cutoff is 0.9; calibration coverage at that cutoff is 0.9005 and evaluation coverage is 0.90. The block standard errors are approximately:
- block length 25: 0.02726;
- block length 50: 0.02226;
- block length 100: 0.03147.

The iid reference standard error is approximately 0.00949. This deterministic fixture makes the block-vs-iid accounting visible; it is not an empirical confidence interval study or a physical PMU result.

Twenty-two authored Rust tests cover the known answer, FDM tail arithmetic, block sensitivity, assumption/identity gates, not-bounded mode, invalid envelopes, group overlap, insufficient blocks and explicit non-anytime-valid claim limits.

## 6. Run

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-temporal-dependence-audit \
  --example hardsec_temporal_dependence_audit

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-temporal-dependence-audit \
  --example hardsec_temporal_dependence_audit -- \
  --model tests/fixtures/hardsec/temporal-dependence-v1/model.json \
  --data tests/fixtures/hardsec/temporal-dependence-v1/data.json \
  --receipt /tmp/hardsec-022-temporal-dependence-receipt.json
```

## 7. Benchmark programme

- iid-score sanity case;
- AR(1), finite-memory MA and nonlinear causal recursions;
- overlapping windows and persistent scheduling phases;
- long-memory Gaussian linear processes, including regimes where root-n normality fails;
- model mismatch and deliberately underestimated FDM envelopes;
- score-indicator dependence versus raw-score ACF;
- block lengths and block count sensitivity;
- iid standard error versus block-based standard error;
- FDM marginal coverage bound versus empirical undercoverage in simulation;
- single-start/restarted CCTM and episode-level decisions on identical source traces.

For every run, preserve score/model digests, exact calibration/evaluation execution IDs, device boots, block lengths and data split. Calibration windows from one execution do not become independent innovations by assigning them separate IDs.

## 8. Integration and claim ceiling

The receipt can be attached to HARDSEC-021 provenance as a dependence diagnostic. A passing geometric envelope only means the supplied per-lag numbers fit the supplied envelope. It does not prove:
- the true PMU process has the declared causal innovation representation;
- its innovations are iid;
- its score-threshold FDM is bounded as claimed;
- stationarity or continuity assumptions hold;
- accepted telemetry remains a valid selected sequence;
- the restarted CCTM is a supermartingale under this dependence model.

The tool therefore hard-codes `sequential_ctm_validity_inferred=false`. A separate theorem/implementation audit is required before any sequential PFA guarantee can use this dependence contract. Until then, block-based uncertainty and independent-execution/episode-level evaluation are diagnostics, not replacements for proof.
