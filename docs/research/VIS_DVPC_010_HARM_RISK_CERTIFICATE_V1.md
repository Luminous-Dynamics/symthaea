# VIS-DVPC-010: Finite-Sample Harm-Rate Certificate v1

**Status:** Draft offline certification reference.  
**Issue:** [#8550](https://github.com/Luminous-Dynamics/symthaea/issues/8550)  
**Calibrated decision router:** [PR #8532](https://github.com/Luminous-Dynamics/symthaea/pull/8532)  
**Safety/tail oracle:** [PR #8542](https://github.com/Luminous-Dynamics/symthaea/pull/8542)  
**Paired-trace scorer:** [PR #8506](https://github.com/Luminous-Dynamics/symthaea/pull/8506)

## Research gap

A finite paired-trace selector can keep its observed violation count below a deterministic cap while still having weak evidence about the population harm rate. A calibrated predicted-benefit lower bound is also not, by itself, a bound on realized harmful-escalation probability.

Prior art points to finite-sample risk control:
- [Conformal Policy Control (ICML 2026)](https://proceedings.mlr.press/v306/prinster26a.html) uses a safe reference policy to regulate optimized behavior under a declared risk tolerance.
- [Fixed-Sequence Confidence-Interval Calibration for Finite-Sample Selective Risk Control (published Oct 2, 2026)](https://www.mdpi.com/2571-905X/9/5/109) uses one-sided confidence intervals to certify thresholds in a fixed sequence under IID calibration/deployment assumptions.
- [High Probability Risk Control Under Covariate Shift (2025)](https://proceedings.mlr.press/v266/almeida25a.html) addresses the gap between identically distributed calibration data and covariate-shifted target data.

This prototype implements a conservative exact-binomial certificate, not either paper's full algorithm.

## Statistical contract

Source: `tools/vision-dvpc-harm-risk-certificate.rs`

For a frozen policy, only escalations enter the denominator of the harmful-escalation rate. Every evaluated escalation must have an observed binary post-commit harm label. Non-escalations must not carry a harm label. Policy identity/digest, loss profile, calibration split, domain and source-group IDs are validated before any statistics are calculated.

For (k) harmful escalations among (n) selected escalations, the tool computes a one-sided Clopper–Pearson upper confidence limit (p_U) by solving

[
Pr_{p_U}[X le k] = delta_{mathrm{adj}}, qquad Xsimmathrm{Binomial}(n,p_U)
]

for (k<n), with (p_U=1) when (k=n). In the zero-harm case,

[
p_U = 1-delta_{mathrm{adj}}^{1/n}.
]

The implementation uses log-binomial probabilities and bisection for nonzero harm counts; it bounds sample size to 100,000 per certificate.

To preserve family-wise error probability across the predeclared policy family, overall rate, and each predeclared group, it uses Bonferroni allocation:

[
delta_{mathrm{adj}}=
rac{delta_{mathrm{family}}}
{N_{mathrm{policies}},(1+N_{mathrm{groups}})}.
]

This is intentionally conservative. The overall rate **and every predeclared source group** must have an upper bound at or below the risk threshold. If a group has no evaluated escalations, the result is `INSUFFICIENT_EVIDENCE` rather than 0% risk. A nonzero but weak sample whose upper bound exceeds the target yields `NOT_CERTIFIED_RISK_BOUND`.

## What this does not guarantee

The certificate assumes IID/exchangeable held-out outcomes for the exact frozen policy, loss definition, and declared domain. It does not establish safety under arbitrary distribution shift; any calibration row marked `OodSuspected` or `Unknown` is rejected. It does not establish a causal model of harm, safety outside sampled scenarios, or a runtime deployment guarantee.

The policy and threshold must be frozen before the risk-certification split is examined. If a policy is changed after viewing those labels, the same set is no longer an independent certification set; obtain a fresh split or use a valid predeclared multi-policy family with the stated multiplicity correction.

The supplied calibration-manifest digest is retained in the receipt but this standalone module does not compute SHA-256 over the file/trace bytes. A trusted caller must verify exact manifest bytes, dataset rights, split integrity, and policy digest. The evaluator labels must remain inaccessible to the online router.

## Adversarial tests

The thirteen authored Rust tests cover:
1. exact zero-harm bound and the 71-vs-72 example for a 5% target with adjusted (delta=0.025);
2. a known one-harm/ten-escalation Clopper–Pearson reference value;
3. rejection of tiny clean samples that do not certify the population target;
4. zero selected escalations as insufficient evidence;
5. sufficient clean escalations meeting the adjusted target;
6. stronger correction for a larger predeclared policy family;
7. a rare group failing its bound even when the pooled bound passes;
8. a represented group with no escalations returning insufficient evidence;
9. a demonstrated failing group's risk bound taking precedence over another group's insufficient evidence;
10. duplicate trace IDs, malformed labels and stale policy digests;
11. shift states and absent predeclared groups failing closed;
12. deterministic, input-order-invariant receipts bound to policy/manifest metadata;
13. bound monotonicity and the all-harm boundary.

Passing these tests would validate the reference calculations and state machine only, not coverage on real data.

## Benchmark proposal

On frozen, rights-cleared, post-commit paired traces, split data roles explicitly:
- model/utility tuning;
- conformal benefit calibration;
- risk-certificate calibration;
- untouched final evaluation.

Compare all-cheap, all-full, matched-budget random, lightweight threshold, raw-value, calibrated-lower-bound and risk-certified policies. Report jointly:
- certified overall and worst-group harm-rate upper bounds;
- number of traces and selected escalations per group;
- policy acceptance/escalation rate;
- paired decision-utility gain and confidence interval;
- harmful-escalation frequency and tail loss;
- allocator overhead, total latency, energy, abstentions and budget violations;
- in-domain versus shifted/unknown-domain results.

A successful benchmark is not just a low bound: it is improved task utility at matched resource cost with a valid independent risk certificate and no predeclared subgroup failure. Do not reuse the final test split to select a threshold after seeing results.
