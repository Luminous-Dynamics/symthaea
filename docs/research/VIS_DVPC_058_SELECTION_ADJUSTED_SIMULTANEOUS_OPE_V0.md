# VIS-DVPC-058 — Selection-adjusted simultaneous OPE for policy portfolios v0

**Status:** synthetic finite-policy inference benchmark. The max-T multiplier bootstrap is an asymptotic inference comparator and is empirically stress-tested here; this is not a finite-sample coverage theorem.

## Why this is necessary

VIS-DVPC-057 evaluates several policies using one shared reward model and common logged evaluation observations. Selecting the policy with the largest noisy value estimate changes the inferential target: a pointwise 95% interval for each policy does not give 95% simultaneous coverage for the portfolio, and the selected maximum is subject to winner's-curse bias.

Relevant comparators:
- Luo, Fan & Wu, *Simultaneous Statistical Inference for Off-Policy Evaluation in Reinforcement Learning*, NeurIPS 2025, develops simultaneous confidence regions over a continuous initial-state space using sieve Q estimation and multiplier bootstrap: https://proceedings.neurips.cc/paper_files/paper/2025/hash/65f02fa90471d4b61f7c327f33f2b92e-Abstract-Conference.html
- Dann, Ghavamzadeh & Marinov, *Multiple-policy High-confidence Policy Evaluation*, AISTATS 2023, reuses shared samples to evaluate multiple policies with guarantees depending on policy overlap: https://proceedings.mlr.press/v206/dann23a.html
- Cief, Kveton & Kompan, *Cross-Validated Off-Policy Evaluation*, AAAI 2025, studies OPE estimator selection and tuning: https://ojs.aaai.org/index.php/AAAI/article/view/33765
- Berman, Zhang & Zhao, *Valuing Winners*, JSM 2026, distinguishes global winner's curse, selective winner's curse, coverage and deployment regret, and reports that different corrections trade off differently: https://ww2.amstat.org/meetings/jsm/2026/onlineprogram/abstract.cfm?sid=5610&tid=5613

VIS-DVPC-058 brings the simultaneous-inference and winner's-curse questions into the budgeted, noisy and correlated counterfactual-annotation setting.

## Inference estimand and common-sample design

For policy j, the event-level score is:

\[
\\psi_{i,j}=
\\sum_a\\pi_j(a\\mid X_i)\\widehat q(X_i,a)
+\\frac{\\pi_j(A_i\\mid X_i)}{\\mu(A_i\\mid X_i)}
\\{R_i-\\widehat q(X_i,A_i)\\}.
\]

All K target policies use the same fitted q-hat and the same logged context/action/factual reward/propensity observation. The event-level score vector is therefore joint; no policy-specific resampling is allowed in the multiplier bootstrap.

The q-hat / counterfactual annotation fit stream and evaluation stream use separate deterministic seeds. Conditional on the fitted q-hat and known behavior propensities, the event scores are the iid evaluation observations. The coverage experiment then repeats this complete process over independent outer replicates, so reported coverage includes variation from the independent reward-model fitting stream as well.

## Compared interval methods

1. `pointwise_95_negative_control`: separate nominal 95% two-sided normal intervals for each policy, with no multiplicity correction.
2. `bonferroni_95`: marginal normal intervals at alpha/K, providing a transparent multiplicity-adjusted baseline under its usual marginal interval assumptions.
3. `common_sample_multiplier_max_t_95`: studentized Rademacher multiplier bootstrap of the maximum absolute policy statistic. One multiplier is drawn per logged observation and applied to every policy score for that observation, preserving estimated cross-policy dependence.

For score rows `psi_i` and sample mean `mean_j`, the max-T bootstrap draws a shared multiplier `xi_i` and computes:

\[
T^*=\\max_{j\\leq K}
\\left|
\\frac{N^{-1}\\sum_i\\xi_i(\\psi_{i,j}-\\bar\\psi_j)}{SE_j}
\\right|.
\]

The empirical 95th percentile is used as a common critical value. The implementation uses 199 bootstrap multipliers per outer replicate in v0. This is a deliberately small reproducible benchmark setting; production-grade quantile stability must be checked with larger bootstrap budgets.

## Winner certification and abstention

The selected policy is `argmax_j mean_j`. The rule certifies policy j only when its simultaneous lower bound exceeds every other candidate's simultaneous upper bound:

\[
L_j>\\max_{k\\ne j}U_k.
\]

Otherwise it abstains. A tie or near-tie should normally lower certification rate, not cause a forced winner. For a true tie, any unique-winner certificate is counted as false certification because no unique best policy exists.

## Decision-relevant metrics

Each method reports:
- simultaneous coverage: probability every policy's true value lies in its interval;
- Wilson 95% interval for the Monte Carlo estimate of simultaneous coverage, showing uncertainty in the benchmark's coverage estimate;
- average marginal coverage across policies;
- winner certification rate, false-promotion probability, and false-promotion probability conditional on certification;
- probability of selecting the truly best policy and expected simple regret;
- global winner's curse: estimated selected-policy value minus the true value of the best policy;
- selective winner's curse: estimated selected-policy value minus the true value of the selected policy;
- average interval width and mean critical value.

The two winner's-curse metrics are intentionally separate: one measures optimism relative to the best achievable policy; the other measures optimism in the selected policy's own value estimate.

## Registered scenarios

- `distinct_policy_gap`: three policies with a materially separated true best;
- `near_tie_policy_gap`: conservative and aggressive policies are moved close to the balanced policy, stressing low-power selection;
- `exact_policy_tie`: all three policies are identical, so a unique-winner certificate must never be issued.

All scenarios use the correlated-source world from VIS-DVPC-056 and the covariance-aware exact worst-policy allocation from VIS-DVPC-057. Inference uses 1,024 independent outer replicates in full mode and 48 in smoke mode; each outer replicate uses 128 logged evaluation observations.

## Acceptance targets

These are research targets, not outcomes claimed in advance:
- simultaneous coverage at least 95% with its Wilson lower bound reported; target lower bound >= 0.94 in the full registered suite;
- false unique-winner promotion <= 5% in exact-tie and near-tie stress cases;
- exact-tie certification rate = 0%;
- the max-T method should preserve joint dependence and must not be implemented as independent per-policy resampling;
- near-tie scenarios should show a measurable abstention/power trade-off rather than forced selection;
- all results deterministic for a fixed source SHA, configuration and seeds; output carries exact code/docs hashes.

The exact finite-sample coverage guarantee is not implied by passing these Monte Carlo thresholds. A separate theory and a broader failure suite are required before calling this safe for consequential policy selection.

## Limits and next steps

v0 assumes iid logged evaluation observations, known positive behavior propensities, fixed target policies, independent q-hat fit/evaluation streams, stationary Bernoulli rewards and a finite K=3 policy family. It does not handle sequential peeking, policy-family growth after seeing evaluation data, propensity estimation, hidden confounding, unbounded return concentration, or continuous policy classes.

Next steps: larger bootstrap budgets (999/4,999), policy-count sweeps K in {3, 10, 50, 100}, overlap and heavy-tail regimes, estimated propensity stress, cross-fitting nuisance estimation, confidence sequences for sequential policy promotion, and matched replication of the NeurIPS 2025 simultaneous OPE procedure where its assumptions apply. Keep annotations out of the factual residual; zero support remains NOT IDENTIFIED.