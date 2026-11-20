# Prior-shift stress test for objective-matched causal probe oracles — 2026-10-23

## Why this experiment

The previous exact-oracle lane compared strategies under one symmetric prior. That is insufficient for an adaptive scientific agent: policy rankings can depend on what was believed before the first probe. This experiment changes prior probabilities over a fixed, tractable SCM game while preserving exact Bellman regret and score-matched oracle comparisons.

## Protocol

- 16 joint hypotheses: chain or fork graph × one of two off-target probabilities (0.1 or 0.6) for each of three attempted targets.
- Three actions, each leading to the intended singleton or an adjacent pair as the realized intervention subset.
- Exact realized-subset telemetry plus the 3-bit SCM outcome.
- Exact Bellman oracles for graph entropy, intervention-kernel entropy and joint graph/kernel entropy.
- 11 policies, three objectives, horizons 1–3.
- Six strictly positive priors: uniform; chain-biased (P(chain)=0.85); fork-biased (P(fork)=0.85); low-spill-biased (each action independently favors low spill at 0.85); high-spill-biased (each action favors high spill at 0.85); and graph–kernel-correlated (P(high spill)=0.2 if chain and 0.8 if fork).
- 594 total policy/objective/horizon/prior rows.

This is a prior-sensitivity audit on a fixed 16-hypothesis game, not a change in graph/mechanism families. Every policy is scored against an oracle minimizing the same terminal entropy objective.

## Key findings at horizon 3

### Joint graph and kernel objective

| Prior | Exact oracle information gain (bits) | Greedy joint regret (bits) | Optimal policy set includes |
|---|---:|---:|---|
| Uniform | 0.941652 | 0.001080 | Exact oracle |
| Chain-biased | 0.801675 | 0.000000 | Greedy joint and exact oracle |
| Fork-biased | 0.801675 | 0.000000 | Greedy joint and exact oracle |
| Low-spill-biased | 0.814777 | 0.000000 | Greedy joint and exact oracle |
| High-spill-biased | 0.613599 | 0.013403 | Exact oracle |
| Graph–kernel-correlated | 0.887614 | 0.002919 | Exact oracle |

Greedy joint information matches the Bellman optimum to numerical tolerance under three of the five non-uniform priors, plus has tiny regret under the uniform prior. But its regret grows to 0.013403 bits under the high-spill prior, where the exact oracle is measurably better. Thus the earlier 0.00108-bit uniform-prior gap is not a universal guarantee.

### Objective-specific policy consistency

At horizon 3, greedy graph information ties the exact graph-entropy oracle under all six tested priors. Greedy kernel information ties the exact kernel-entropy oracle under all six. These are model- and horizon-specific exact findings, not general theorems. The policy ranking for joint uncertainty is more prior-sensitive.

Across the six priors, horizon-three mean and worst regret for greedy policies are:

| Evaluated objective | Greedy policy | Mean regret (bits) | Worst regret (bits) |
|---|---|---:|---:|
| Graph entropy | Greedy graph | 0.000000 | 0.000000 |
| Kernel entropy | Greedy kernel | 0.000000 | 0.000000 |
| Joint entropy | Greedy joint | 0.002900 | 0.013403 |

For comparison, round-robin has mean joint-objective regret 0.108117 bits and maximum 0.188636 bits across these priors; random selection has mean 0.156083 and maximum 0.232837 bits. These are exact expectations in the finite game, not noisy Monte Carlo averages.

## Implications

1. Prior robustness should be a named benchmark dimension. A strategy ranking obtained under a uniform prior should not be generalized until graph-dominant, mechanism-dominant, and correlated priors have been tested.
2. Objective-matched regret remains necessary. Graph and kernel policies look optimal for their matching objectives here, but they incur regret against the joint objective. Metrics must stay separated.
3. The high-spill prior is a useful adversarial case. It raises greedy-joint regret from 0.001080 bits under uniform belief to 0.013403 bits. The next work should find more severe prior/model cases and study whether limited lookahead gives a stable improvement.
4. This is not external qualification. Only chain/fork graphs, two spill probabilities, exact subset telemetry, and three-probe horizons are included.

## Research and industry comparison

Action-dependent random target-subset distributions are established prior art: Choo, Shiragur and Uhler, “Causal Discovery under Off-Target Interventions,” AISTATS 2024. ACDB (September 2026) evaluates budgeted active causal discovery with separate structure and efficiency metrics. CausalArena (September 2026) highlights ranking shifts across SCM families and evaluation protocols. TCD-Arena (ICLR 2026) stress-tests methods against assumption violations. This prior-shift experiment complements those concerns but does not run their methods or share their full contracts. No external-baseline or SOTA claim is made.

References:
- Choo et al. (AISTATS 2024): https://proceedings.mlr.press/v238/choo24a.html
- ACDB (September 2026): https://arxiv.org/abs/2609.31675
- CausalArena (September 2026): https://arxiv.org/abs/2609.11897
- TCD-Arena (ICLR 2026): https://tcd-arena.github.io/

## Reproduction and evidence

Run from the repository root:

PYTHONPATH=validation python3 -m unittest -v test_prior_shift_oracle_benchmark

PYTHONPATH=validation python3 prior_shift_oracle_benchmark.py --horizon 3 --source-git-blob 5222bd65b53cb872cdef04d34247f395b825c1ee --output validation/prior_shift_oracle_results_exact_2026-10-23.json

The full exact artifact contains 594 rows, is 234,478 bytes, and has SHA-256 a03028bc12ab4fce908f4824add119e07e246826367658884cf1eca27da30f2a. The executed source Git blob is 5222bd65b53cb872cdef04d34247f395b825c1ee; the test Git blob is be7a515cb3cc8a3e7496e24141e2f45d48508eeb. The new tests pass 6/6 locally and the five-suite combined test command passes 36/36. These are local results, not remote CI status.
