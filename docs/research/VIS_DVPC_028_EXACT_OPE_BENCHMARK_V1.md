# VIS-DVPC-028 — Exact finite-sample OPE benchmark v1

**Status:** standalone synthetic known-answer experiment. No real-trace, production-safety, or SOTA claim.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8730  
Structural-support preflight: https://github.com/Luminous-Dynamics/symthaea/pull/8729

## Paradigm shift

An off-policy evaluator should prove which claims are identified before assigning a numeric winner. With selective feedback, only the action chosen by the logger reveals a reward. If a target policy assigns positive probability to an action with zero logging probability, that target value cannot be identified from these logs alone. The benchmark emits `NOT_IDENTIFIED` and suppresses estimates for that scenario.

For supported scenarios, this tool exactly enumerates every length-6 logged sample under a stationary two-action Bernoulli bandit model. It computes expected estimator value, bias, variance, RMSE and maximum absolute estimation error over the exact sampling distribution. It is enumeration—not Monte Carlo—so these finite-sample summaries have no simulation noise under the declared DGP. The output is TSV with one metric per row and explicit record-type, scenario, estimator, verdict, metric, and value fields.

## Estimators

- **DM:** direct method using the fixed predeclared reward model `qhat`.
- **IPS:** inverse propensity score using the exact declared logging propensities.
- **SNIPS:** self-normalized IPS, normalized by the observed importance-weight sum.
- **DR:** doubly robust/AIPW form: `q_target + weight * (reward - qhat_selected)`.
- **CLIPPED_IPS:** IPS with the predeclared cap applied to importance weights.
- **SWITCH:** direct-model prediction plus a DR correction only when the weight is at or below the predeclared cap.

For this controlled experiment, the logging propensity and Bernoulli DGP are known exactly to the evaluator. This deliberately favorable condition establishes reference behavior before nuisance-model estimation is added. The estimates are diagnostic outputs, never authorization evidence.

## Frozen DGP

`tests/fixtures/vision-dvpc-028-scenarios.tsv` defines six scenarios: healthy overlap with correct/misspecified reward models; logging propensity A of 0.9, 0.95, and 0.99; and exact-zero support. Each supported scenario enumerates all `4^6 = 4096` possible logged action/reward sequences, with exact probability mass. Rewards are independent Bernoulli conditional on the selected action; the benchmark does not model contexts, unobserved confounding, clustering or adaptive logging in v1.

The target policy chooses A with probability 0.5. Reward means are 0.2 for A and 0.8 for B, so the evaluator-known target value is 0.5. The logging policy is the scenario's A probability. The SWITCH/clipping cap is frozen at 2.5. The deliberately misspecified reward model differs from the true means.

## Run

```sh
rustc --edition=2024 --test tools/vision-dvpc-ope-exact.rs -o /tmp/vis-dvpc-028-tests
/tmp/vis-dvpc-028-tests
rustc --edition=2024 -O tools/vision-dvpc-ope-exact.rs -o /tmp/vis-dvpc-028
/tmp/vis-dvpc-028 --sweep
```

## Independent numerical known answers

An independent exact-enumeration reference calculation for `weak_overlap_001_misspecified` gives IPS variance `3.300084175084`, SNIPS expected estimate `0.233448729630`, clipped-IPS expected estimate `0.120000000000`, and SWITCH expected estimate `0.350000000000` with variance `0.006743476431`. The Rust tests pin these values within a declared `1e-8` tolerance.

## Interpretation boundaries

- IPS expectation is correct under the exact declared propensity and support assumptions; its variance can grow rapidly under poor overlap.
- DR can have correct expectation under correct logging propensities even when the reward model is misspecified, but its variance is not universally small.
- DM bias reflects reward-model misspecification.
- Clipped IPS and SWITCH deliberately expose a bias/variance trade-off; a lower variance is not automatically a more valid safety claim.
- SNIPS can be biased at finite sample sizes.
- `NOT_IDENTIFIED` is not a numerical estimate and must not be overwritten by a direct-model prediction.
- No confidence interval, coverage, clustered-data, adaptive-collection or target-shift guarantee is produced by v1.

## Research anchors

- Wang, Agarwal & Dudík, *Optimal and Adaptive Off-policy Evaluation in Contextual Bandits*, ICML/PMLR 2017. It studies minimax error and shows how IPS/DR can be high-variance and how SWITCH can improve the bias–variance trade-off: https://proceedings.mlr.press/v70/wang17a.html
- Su et al., *Doubly Robust Off-policy Evaluation with Shrinkage*, ICML/PMLR 2020: https://proceedings.mlr.press/v119/su20a.html
- Mukherjee, Hanna & Nowak, *SaVeR: Optimal Data Collection Strategy for Safe Policy Evaluation in Tabular MDP*, ICML/PMLR 2024: https://proceedings.mlr.press/v235/mukherjee24a.html

These are methodological baselines, not directly comparable scoreboards. The next version should add propensity and outcome-model misspecification grids, support-aware intervals, and cluster/adaptive designs before any real-trace claim.
