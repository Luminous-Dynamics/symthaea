# VIS-DVPC-054 — Budgeted multi-source counterfactual annotation for OPE v0

**Status:** synthetic exact-objective benchmark and acquisition-algorithm falsification harness. Not a clinical validation, public-dataset result, SOTA claim, or deployment certificate.

## Motivation

VIS-DVPC-053 studies reward-model shrinkage and cross-fitted DR. VIS-DVPC-054 asks which missing counterfactual measurements are worth acquiring, from which source, at what price, and in which context-action cell.

Two 2026 papers motivate this work:
- Mandyam et al., CANDOR (CHIL 2026 / PMLR 333:644–687), report that imperfect expert counterfactual annotations can worsen ordinary OPE. Their annotated DR construction uses annotations in the reward-model/DM component while preserving the factual logged reward in the importance-sampling residual: https://proceedings.mlr.press/v333/mandyam26a.html
- Xiang et al., Budgeted Multi-Source Counterfactual Annotation for OPE, arXiv:2610.10974 (2026-10-07), formulate heterogeneous annotation allocation as an integer optimization problem, derive a first-annotation threshold and four local annotation-value regimes, and propose a majorization-minimization allocator with DP subroutines: https://arxiv.org/abs/2610.10974

The 2026 paper reports scaled-MSE reductions of 20.58% and 10.77% in two fixed-profile experiments, and a mean 0.05% optimality gap (maximum 0.26%) for a greedy-initialized MM method on 30 small certified instances. Those are published-paper results, not results reproduced here. Our first goal is an independently auditable small-instance baseline before attempting a faithful MM replication.

## Data protocol

The standalone implementation is tools/vision-dvpc-054-budgeted-annotation-ope.rs.

For each context-action cell (x,a), the frozen synthetic world supplies factual fit count nf, Bernoulli reward mean r and variance sigma-squared, target-policy mass pi(a|x), behavior propensity mu(a|x), source-specific annotation bias epsilon_k(x,a), excess variance Delta_k(x,a), unit cost c_k, and capacity.

The source profiles, factual fit data, annotation observations, and final OPE evaluation stream are distinct. Monte Carlo model fitting and final evaluation use independently derived deterministic seeds. The plugin-profile plan is selected using a deliberately shifted source profile and evaluated under the hidden true profile.

v0 uses two contexts, three actions and three annotation sources, with a global integer token budget and per-context annotation caps. Source quality varies by cell; the best source can therefore change across context/action pairs.

## Estimator boundary: annotations improve q, not the factual residual

The estimator is DM+IS / annotated DR:

\[
\widehat V(\pi)=\frac1N\sum_{i=1}^N\left[
\sum_a\pi(a\mid X_i)\widehat q(X_i,a)
+\frac{\pi(A_i\mid X_i)}{\mu(A_i\mid X_i)}
\{R_i-\widehat q(X_i,A_i)\}\right].
\]

Counterfactual annotations may improve q-hat. They do not replace the factual logged reward R, do not replace the exact behavior propensity mu, and do not establish factual action support. The importance-sampling residual still uses the action actually selected by the logger.

A zero behavior propensity for an action with positive target-policy mass remains NOT IDENTIFIED. An annotation count cannot silently change that verdict.

## Exact allocation objective

For a cell with nf independent factual fit observations and nk annotations from source k, let n = nf + sum_k nk. Under independent source errors:

\[
B_{xa}=\frac{\sum_k n_k\epsilon_k(x,a)}{n},
\qquad
W_{xa}=\frac{n_f\sigma_{xa}^2+
\sum_k n_k\{\sigma_{xa}^2+\Delta_k(x,a)\}}{n^2}.
\]

If n=0, the registered default reward estimate is used; in the current worlds factual fit counts are positive in every cell.

For a fixed context, after integrating over reward-model fit noise and the factual logged action/reward, the exact variance of the annotated DM+IS score is:

\[
\begin{aligned}
\operatorname{Var}(Z\mid X=x)
={}&\sum_a\frac{\pi_a^2}{\mu_a}\sigma_a^2
+\sum_a\pi_a^2\left(\frac1{\mu_a}-1\right)W_a\\
&+\sum_a\frac{\pi_a^2}{\mu_a}B_a^2
-\left(\sum_a\pi_a B_a\right)^2.
\end{aligned}
\]

The final term matters: source biases can cancel after target-policy weighting. Treating source bias magnitudes as independent penalties misses this interaction.

Let Vx = sum_a pi(a|x) r(x,a). The full per-event variance also includes between-context variation:

\[
\operatorname{Var}(Z)=
\operatorname{Var}_{X}(V_X)+
\mathbb E_X[\operatorname{Var}(Z\mid X)].
\]

For N independent evaluation events, exact expected MSE is Var(Z)/N. This is a fixed-profile objective under independent-error assumptions, not a universal formula for correlated annotators or adaptive acquisition.

## Optimizers and certification

The exact small-instance solver enumerates all feasible source-count allocations within each context, computes each option's exact risk component, then combines context options by dynamic programming under the shared budget. It reconstructs and rechecks the chosen integer allocation. This is a global optimum for the finite registered instance, up to floating-point comparison tolerance; it is not a scalable optimizer claim.

Compared methods: no annotations; greedy maximum marginal MSE gain; greedy marginal gain per cost unit; best single-source allocation; pairwise-exchange local search from max-gain greedy; exact DP oracle using true source profiles; exact DP using a deliberately misspecified plugin profile, evaluated under the true profile.

Pairwise exchange is a local-search baseline, not a reproduction of Xiang et al.'s MM algorithm. No SOTA claim is made.

## First-annotation regimes

For factual sample size nf, reward variance sigma-squared, one-source excess variance Delta, squared annotation bias epsilon-squared, and eta=1-mu, the first-annotation condition is:

\[
\epsilon^2\leq
\eta\left[\frac{n_f+1}{n_f}\sigma^2-\Delta\right].
\]

Five frozen rows cover the four local regimes: (I) annotations harmful; (II) finite interior batch is best; (III) every feasible addition helps; (IVa) the first annotation hurts but a sufficiently large batch wins; (IVb) the first annotation hurts and the largest batch still loses. Tests check the threshold against the exact one-step objective difference and verify each registered regime's shape.

## Independent Monte Carlo and negative control

The benchmark compares its analytic MSE with an independent Monte Carlo fitting/evaluation pipeline using separate seeds. Monte Carlo is a falsification check, not the optimizer's objective.

A deliberately wrong control substitutes an annotation from an unchosen action/source into the factual importance-sampling residual. That is not the estimator under study; its bias is reported separately to make the boundary executable.

## Output and reproducibility

The executable emits 14 RESULT rows (two worlds × seven plans), two NEGATIVE_CONTROL rows, five LOCAL_REGIME rows, and eight BENCHMARK_CONFIG rows. The focused CI workflow compiles this standalone Rust 2024 source directly, avoiding Cargo workspace metadata; it checks deterministic smoke output, row counts and schemas, full output, and SHA-256 receipts.

## Benchmark targets

- Exact DP agrees with independent exhaustive enumeration on small fixtures.
- Every allocation is feasible; the exact objective is never worse than no annotations under the same true profiles.
- Analytic expected MSE matches independent Monte Carlo within the registered tolerance.
- At least one mixed-source fixture demonstrates opposite-bias cancellation.
- Greedy optimality gaps are reported against the exact DP certificate.
- Plugin allocation is evaluated under hidden true profiles.
- Annotations never bypass support or replace factual residual outcomes.

## Industry and SOTA comparison

CANDOR is the estimator-protocol comparator. Xiang et al. (2026) is the acquisition-objective and optimizer comparator. SCOPE-RL and Open Bandit Pipeline remain useful for matched DM/IPS/DR policy-value evaluation and public logged-bandit parity, but neither substitutes for a heterogeneous counterfactual annotation acquisition benchmark.

A defensible comparison must match the target policy, factual fit data, source costs/capacities, annotation profiles, budget, estimator, and evaluation protocol. It must disclose whether source profiles were known, estimated on independent pilot data, or misspecified. Real data and external validity are not included in v0; check dataset terms and attribution before any download or redistribution.

## Scope limits

The frozen worlds assume independent annotation errors conditional on source and cell, fixed profiles, known behavior propensities, stationary Bernoulli factual rewards and a fixed target policy. They do not establish guarantees for correlated annotators, hidden confounding, nonstationary reward, estimated propensities, or deployment.