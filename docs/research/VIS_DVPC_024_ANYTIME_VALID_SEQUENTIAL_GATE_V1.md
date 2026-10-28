# VIS-DVPC-024: Anytime-valid sequential activation gate v1

**Status:** synthetic exact finite-horizon mechanism benchmark. It is not a production gate and does not establish real-target calibration.

## Why fixed-sample intervals are not enough

A sequential DVPC campaign may inspect results after many observations and stop whenever a favorable result appears. A fixed-sample interval's coverage at one prespecified sample size does not generally protect a procedure that repeatedly checks those intervals and stops at a data-dependent time.

The standalone benchmark tests an explicit alternative: nonnegative e-processes whose probability of ever crossing a threshold is controlled over the full horizon under the declared null. The benchmark computes the finite-horizon crossing probability by exact dynamic programming over binomial count states, not Monte Carlo sampling.

## Evidence streams and hypotheses

### Risk stream

Null hypothesis: failure probability p is at least the cap p0=0.05.

For a predeclared alternative q<p0, the likelihood-ratio process after n observations and k failures is:

E(n,k;q) = (q/p0)^k ((1-q)/(1-p0))^(n-k).

The scorer mixes four fixed alternatives q in {0.001, 0.01, 0.025, 0.04} with equal weights. Under the composite null p>=p0, every component is a nonnegative supermartingale, so the fixed mixture remains one. The risk claim crosses only when E >= 1/alpha_risk, with alpha_risk=0.025.

### Value stream

Null hypothesis: mean normalized value is at most mu0=0.5, with observations in [0,1].

For each fixed lambda in {0.5, 1.0}, the betting factor is 1 + lambda * (x - mu0). It is nonnegative for x in [0,1] and lambda in [0,1], and its conditional expectation is at most one under the null E[x | past] <= mu0. The scorer mixes the two predeclared lambdas with equal weights. The value claim crosses only when its e-process reaches 1/alpha_value, with alpha_value=0.025.

The total error budget is alpha_risk + alpha_value = 0.05. Both claims must cross before statistical activation; by a union bound, the pair's false-certification probability is at most 5% if the null/conditional-mean assumptions hold. The risk and value streams do not need to be assumed independent for this bound.

## Exact peeking negative control

A deliberately naive baseline reuses a 97.5% one-sided Clopper-Pearson upper bound at sample sizes n=72 and n=120, certifying as soon as any look says its upper bound is at or below 5%. Under the boundary truth p=0.05, the exact probability of ever certifying is approximately 3.2937%, exceeding its nominal per-look alpha=2.5%.

The e-process gate's exact probability of crossing the risk boundary by n=120 under the same p=0.05 truth is approximately 0.8924%, below its 2.5% budget. These are exact finite-horizon values for the fixed benchmark construction, not estimated from random simulation. The e-process gate is more conservative here; compare time-to-certification and power as well as false-certification rate.

## Frozen benchmark and output

The fixture contains 40 scenarios: five horizons {20, 40, 80, 120, 200} crossed with four risk truths {0, 0.05, 0.10, 0.20} and four value means {0.40, 0.50, 0.60, 0.70}. The scorer reports the exact probability of crossing at any time up to each horizon.

The workflow verifies the fixture digest, runs standalone Rust tests, builds the scorer, checks the scenario and output counts, and uploads the machine-readable output plus a receipt binding the commit SHA, Rust version, source hash, fixture hash and output hash.

## Acceptance gates

- The exact finite-horizon probability of crossing the risk gate is at most alpha_risk under p=p0 and representative unsafe probabilities p>p0, for every frozen horizon.
- The exact finite-horizon probability of crossing the value gate is at most alpha_value under means at or below mu0, for every frozen horizon.
- The predeclared repeated fixed-n interval baseline exceeds its nominal per-look error on the known boundary fixture, while the e-process crossing rate remains within its budget.
- Positive-control power is nontrivial: the gate must not behave as an always-deny system.
- Inputs, alternatives, betting fractions, look schedule, alpha allocation and output are frozen and reproducible.
- Every result is scoped to the declared iid Bernoulli streams; value stream validity extends to bounded observations only under the corresponding conditional-mean assumption.

## Research context

- [Anytime-Valid Inference for Online Ranking of LLMs (ICML/PMLR 2026)](https://proceedings.mlr.press/v306/gu26l.html) gives an e-process framework for sequential ranking with family-wise error control under adaptive monitoring.
- [Anytime-Valid Evidence for Prespecified Predictive Corrections (2026)](https://arxiv.org/abs/2608.08174) studies sequential evidence for a declared predictive correction under source-to-target change and emphasizes that evidence for a correction does not uniquely identify the shift mechanism.
- VIS-DVPC-023 checks fixed-sample marginal and joint coverage on an exact Bernoulli grid; VIS-DVPC-024 addresses repeated inspection/stopping, a separate property.

## Limitations

This benchmark does not prove correctness under arbitrary adaptive data selection, clustered dependence, nonstationary targets, unmodeled target shift, multiple subgroups, or reuse across multiple campaigns. Real use needs a conditional null that remains true under the acquisition policy, appropriate cluster-level procedures, a preregistered alpha allocation across claims and slices, independently verified evidence receipts, and an untouched target confirmation set. The target-calibration and cost/provenance gates remain outside this mathematical scorer.

No SOTA, real-world safety, calibration, transportability or production-authority claim follows from these synthetic cases.
