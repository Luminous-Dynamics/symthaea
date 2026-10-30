# VIS-DVPC-057 — Shared counterfactual evidence for multiple target policies v0

**Status:** synthetic finite-instance portfolio benchmark. No external-data, confidence-coverage, clinical, or deployment claim.

## Research motivation

VIS-DVPC-054–056 select counterfactual annotations for one target policy. Production OPE often evaluates a family of candidate policies and selects among them. A single-policy allocation can improve the nominal target while leaving another candidate poorly evaluated; selecting the maximum of noisy estimates can then amplify evaluation error.

Recent comparators:
- Dragazis, Chen & Pacchiano, *Explore Once, Evaluate Many: Multiple Policy Evaluation for Stochastic Linear Bandits* (arXiv:2610.22252, 2026-10-15) propose PRISMA: one shared information design and one shared ridge reward model for K target policies, with logarithmic explicit dependence on K under its known-feature and coverage assumptions: https://arxiv.org/abs/2610.22252
- Liu, Chen & Zhang, *Efficient Multi-Policy Evaluation for Reinforcement Learning* (AAAI 2025) design one behavior policy to reduce variance across a set of target policies: https://ojs.aaai.org/index.php/AAAI/article/view/34086
- Weissmann et al., *Barycenter Policy Design for Multiple Policy Evaluation* (EWRL 2025) study overlap-aware behavior policy design and best-policy selection: https://openreview.net/forum?id=z7D9OlRYHg
- Dann et al., *Multiple-policy High-confidence Policy Evaluation* (AISTATS 2023) study simultaneous high-confidence evaluation using shared samples: https://proceedings.mlr.press/v206/dann23a.html

These methods motivate shared evidence and policy overlap. VIS-DVPC-057 asks the complementary question: how should a limited budget of imperfect/correlated counterfactual annotations be allocated to one reward model that will serve an entire policy portfolio?

## Frozen policy portfolio

Three fixed policies share one annotation-augmented reward model:

- `conservative`: context-dependent distribution concentrated on a lower-variance action profile;
- `balanced`: the VIS-DVPC-054 nominal target policy;
- `aggressive`: context-dependent distribution concentrated on a higher-upside action profile.

The priors are `[0.25, 0.50, 0.25]`. The portfolio shares factual fit data, counterfactual annotations, and the same logged OPE evaluation sample. Each policy is evaluated with the same q-hat and the same factual residual data. Consequently, policy-value estimates are statistically dependent.

## Acquisition objectives

Let `MSE_j(n)` be the exact annotated-DR / DM+IS mean squared error for portfolio policy j under the shared allocation n.

**Prior-weighted risk:**

\[
J_{prior}(n)=\\sum_{j=1}^{K}w_j\\,MSE_j(n).
\]

**Worst-policy risk:**

\[
J_{max}(n)=\\max_{j\\in[K]}MSE_j(n).
\]

The exact finite-fixture oracle enumerates the feasible annotation allocations for context 0 and context 1, combines them under the shared cost budget, and evaluates all three target policies for every feasible combination. It independently certifies the optimum for each scalar objective. This is deliberately a small-instance exact oracle, not a scalable optimizer claim.

Compared plans:
- exact allocation for the nominal balanced policy only;
- exact prior-weighted portfolio allocation;
- exact worst-policy portfolio allocation;
- greedy marginal worst-policy-risk allocation.

The exact solver enumerates options per context and exhaustively combines them. It does not call the single-policy dynamic program and relabel that result as portfolio-optimal.

## Joint Monte Carlo protocol

For each replicate, draw one annotation-augmented reward model q-hat, then draw one logged evaluation dataset. Reuse that exact q-hat and the same contexts, logged actions, factual rewards and propensities to compute all K policy estimates.

This estimates the joint covariance matrix:

\[
\\Sigma_{ij}=Cov(\\widehat V(\\pi_i),\\widehat V(\\pi_j)).
\]

Report policy-specific bias, variance and MSE; cross-policy covariance; probability of selecting the true best policy; and expected simple regret:

\[
Regret=V(\\pi^*)-V(\\widehat j),
\\qquad
\\widehat j=\\arg\\max_j\\widehat V(\\pi_j).
\]

An empirical covariance matrix computed from common replicate vectors must be PSD up to numerical tolerance. Marginal MSE or intervals alone do not certify simultaneous coverage.

## Correlation integration

The benchmark inherits the low-rank PSD shared annotation error model from VIS-DVPC-056. Every target policy sees the same q-hat, so its reward-model uncertainty is not resampled per policy. The three covariance stress worlds are: within-family panel dependence, shared rubric plus multiple family dependence, and broad common-mode dependence.

All target policies have common support under the factual logger. Annotations only affect q-hat; they never replace the factual logged reward or behavior propensity. Zero logger support remains NOT IDENTIFIED.

## Metrics

Each `PORTFOLIO_RESULT` reports the scenario/method, budget/cost and total annotations, prior-weighted and worst-policy analytic MSE, policy-specific analytic MSEs, common-sample Monte Carlo biases/variances/MSEs, cross-policy covariances, best-policy selection accuracy, expected simple regret, source counts, and gaps to both exact portfolio objectives.

## Acceptance gates

- Portfolio prior weights and target policy distributions validate; invalid weights/policies fail closed.
- Exact prior and max-risk oracles match independent exhaustive enumeration.
- Prior-exact risk is no worse than the single-policy baseline under the same prior-weighted objective; minimax-exact risk is no worse under its own worst-policy objective.
- The shared-sample covariance matrix is PSD within numerical tolerance.
- Empirical policy MSEs track the exact per-policy MSEs within a registered tolerance across independent replicates.
- Best-policy selection accuracy and expected simple regret are deterministic for fixed seeds and use no tuning/evaluation sample reuse.
- The benchmark includes three dependence scenarios and four acquisition plans per scenario.
- No simultaneous confidence-coverage claim until a calibrated joint high-confidence method is added.

## Industry comparison and stretch targets

PRISMA is the structural comparator for shared information across K policies in linear reward models. AAAI 2025 multi-policy evaluation and EWRL 2025 barycenter design are the behavior-policy comparators. Dann et al. 2023 is the high-confidence simultaneous-evaluation comparator. The present synthetic tabular benchmark differs in its explicit noisy/correlated counterfactual annotation acquisition and therefore is not a direct reproduction of those methods.

Stretch goals: policy-set barycenter/second-moment envelope; feature-structured shared reward model; adaptive policy queries; sequential annotation acquisition based on expected reduction in joint selection regret; and anytime-valid simultaneous policy confidence sequences. The goal is to optimize the decision made after evaluation, not just the marginal error of one preselected target.