# Exact prior-misspecification benchmark — 2026-10-24

## Why this lane exists

The previous prior-sensitivity experiment changed the prior used by both the world and the learner. That does not test misspecification. This benchmark separates the true posterior \`q\` (which determines how observations are generated) from the learner's posterior \`p\` (which determines action choice and Bayesian updating). It scores the learner against the data-generating distribution, rather than its own uncertainty alone.

## Protocol

The exact toy model has 16 hypotheses: chain/fork graph structure crossed with three binary action-specific off-target spill parameters (0.1 or 0.6). We use six strictly positive priors: uniform, chain-biased, fork-biased, low-spill-biased, high-spill-biased and graph/kernel-correlated. Observations include the exact realized intervention subset and a three-bit outcome. Policies include round-robin, random, graph/kernel/joint greedy information, coverage variants, learner-prior entropy oracles, and a cross-entropy Bellman oracle.

For graph, kernel assignment and joint prediction separately, the scores are expected terminal cross-entropy (log loss) and Brier risk. The cross-entropy oracle knows both \`q\` and \`p\` when choosing actions; it is an **unattainable diagnostic upper bound**, not an implementable policy when the true prior is unknown.

The complete 6 × 6 true-prior/learner-prior grid was run for horizons 1–2 (2,592 rows). A full grid at horizon 3 exceeded the local execution limit and is **not** claimed as complete. A separate horizon-3 boundary slice tests uniform and high-spill-biased priors in both directions, including matched controls (432 rows across horizons 1–3).

## Results

### Full prior-pair grid, horizon 2

Across the 30 misspecified pairings, excluding the diagnostic true-prior-aware oracle, mean excess cross-entropy against that oracle was:

| Prediction scored | Mean excess (bits) | Worst excess (bits) | Worst pair / policy |
|---|---:|---:|---|
| Graph | 0.09657 | 0.42626 | chain true / fork assumed / random |
| Kernel | 0.05665 | 0.38914 | low-spill true / graph-kernel-correlated assumed / greedy graph |
| Joint | 0.07300 | 0.39050 | chain true / fork assumed / random |

For the joint score, greedy-joint selection had mean / worst excess 0.02929 / 0.15437 bits; random selection had 0.19493 / 0.39050 bits. These are comparisons inside this small exact model, not general superiority claims.

**Evidence can temporarily worsen predictions under prior misspecification:** six of 1,080 misspecified horizon-2 policy/objective rows had negative cross-entropy reduction versus the learner's initial forecast. All six were graph-score cases. The worst was high-spill truth with a graph/kernel-correlated learner prior and random probing: log-loss worsened by 0.07321 bits and Brier risk worsened by 0.03259. This is a finite-budget failure mode; it does not establish asymptotic inconsistency in this finite-support model.

### Horizon-3 boundary slice

Under the two mismatched pairings in this slice, joint-score greedy-joint had mean / worst excess cross-entropy 0.04960 / 0.05557 bits. Random selection had 0.21556 / 0.23023 bits. With only two mismatched pairs, this is a sanity check, not a broad robustness result.

## Paradigm shift and next gate

Evaluate **decision quality under distribution shift**, not just information gain under a correctly specified prior. A serious system should expose prior sensitivity, detect when fresh evidence degrades proper scores, and select actions robust to a declared ambiguity set. The next experiment should implement a deployable minimax-regret or worst-case expected-score policy over plausible priors; the current oracle is only a ceiling.

Then expand beyond chain/fork to collider/reverse-chain, nonlinear and temporal mechanisms, confounding, noisy/missing target-subset telemetry, intervention cost and task disturbance, and explicit stopping. Compare at equal budgets with structural F1/SHD, calibration/log loss/Brier, kernel L1/KL, action cost, runtime, memory, and per-family uncertainty intervals.

## Reproducibility

- Source \`validation/prior_misspecification_benchmark.py\`: Git blob \`541c31521e1b9e49dfb84c0fa265e877ca9ea632\`; SHA-256 \`c83c16e51aba4c184a2da8d90c9dfad0f5bfae40efa6f4b27cfcb283717ea84f\`.
- Tests \`validation/test_prior_misspecification_benchmark.py\`: Git blob \`c28fc1573a3bb982cc881a75018d931c49f492dd\`; SHA-256 \`33c3fdb75d36bb6b5f22d3a7c3bdf443392517fc9d3ff799c6bb4d7978bc5106\`; 7/7 dedicated tests pass.
- All six benchmark test modules: **43/43 local tests passed**.
- Full-grid exact JSON: 1,505,536 bytes; SHA-256 \`aa926e0cac8d5d45efcfaf94354d073e63c36e500f0c03e7274f4268608e2789\`.
- Horizon-3 boundary exact JSON: 247,420 bytes; SHA-256 \`c8968415d6ce633f5e8984f98333fdeb07b6850e700bb3addb3c0897e9db6f63\`.
- Exact result files are local-only; compact summary and receipt are committed separately. Results bind to the source blob above.

## External comparison boundary

Prior art already studies action-dependent off-target interventions (Choo et al., AISTATS 2024). The newer ACDB benchmark separates skeleton recovery, directed-DAG recovery and intervention efficiency; CausalArena varies synthetic, semantic, formula-grounded and real-data regimes; TCD-Arena evaluates time-series discovery under 33 assumption violations. Those designs motivate our next gates, but **none was run here**, and their F1/SHD values cannot be numerically compared to this toy cross-entropy study.

- [Choo et al., Causal Discovery under Off-Target Interventions](https://proceedings.mlr.press/v238/choo24a.html)
- [ACDB](https://arxiv.org/abs/2609.31675)
- [CausalArena](https://arxiv.org/abs/2609.11897)
- [TCD-Arena](https://tcd-arena.github.io/)

No SOTA, novelty, real-world robustness, or production-readiness claim is made.
