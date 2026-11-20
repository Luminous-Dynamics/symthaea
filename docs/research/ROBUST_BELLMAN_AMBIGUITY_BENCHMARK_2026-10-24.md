# Multi-step rectangular robust Bellman benchmark — 2026-10-24

## Purpose

The one-step ambiguity-set policy showed an encouraging mean improvement over greedy joint information on selected objectives, but the one-step rule does not plan future probes. This lane implements finite-horizon Bellman recursion under **rectangular ambiguity** and evaluates its induced policy under each fixed true-world prior.

## Policy semantics

The Bellman value minimizes expected terminal cross-entropy over interventions. At every decision node it takes the maximum expectation over currently plausible posterior scenarios; recursion repeats this worst-case step after each observation. Thus an adversary can choose different plausible scenarios at different observation branches. This is dynamically consistent rectangular robust control, but it is more conservative than a model where one unknown true prior is drawn once and stays fixed for the episode.

Comparison policies: the rectangular Bellman policy, the earlier one-step minimax policy, greedy joint information, round-robin, uniform random, and a learner-posterior objective-matched entropy oracle. The evaluator's true-prior label is used only to generate the exact expected score; it is not passed to the robust action selector.

## Exact results

The full 6 true-prior × 6 learner-prior × 3 objectives × 6 policies × horizons 1–2 grid contains **1,296 rows**. The separate horizon-three boundary slice has **216 rows**: two true priors × two learner priors × three objectives × six policies × horizons 1–3.

Mean and worst terminal cross-entropy on the 30 mismatched prior pairs at horizon two (lower is better):

| Objective | Policy | Mean CE (bits) | Worst CE (bits) |
|---|---|---:|---:|
| Graph | Rectangular Bellman | 1.020183 | 1.642688 |
| Graph | One-step minimax | 0.858556 | 1.609473 |
| Graph | Greedy joint | 0.897715 | 1.708013 |
| Graph | Round-robin | 0.994745 | 1.866977 |
| Graph | Random | 1.079059 | 2.035729 |
| Kernel | Rectangular Bellman | 3.177572 | 6.100158 |
| Kernel | One-step minimax | 3.168323 | 6.075885 |
| Kernel | Greedy joint | 3.194939 | 6.177194 |
| Kernel | Round-robin | 3.147793 | 6.075885 |
| Kernel | Random | 3.163253 | 6.109654 |
| Joint | Rectangular Bellman | 4.265697 | 6.917406 |
| Joint | One-step minimax | 4.188293 | 6.785929 |
| Joint | Greedy joint | 4.205072 | 6.917406 |
| Joint | Round-robin | 4.263053 | 6.824400 |
| Joint | Random | 4.370710 | 6.943428 |

This is a meaningful negative result: the rectangular Bellman policy is more conservative, but it is **worse than the one-step minimax policy on mean graph, kernel and joint cross-entropy** under the evaluation distribution where each true prior stays fixed. It beats random on mean graph and joint scores, but that is not sufficient evidence of an advantage over stronger policies.

The horizon-three boundary slice checks only the two mismatched prior directions between uniform and high-spill-biased priors. At that slice, rectangular Bellman mean terminal CE is 0.817997 (graph), 3.098311 (kernel) and 3.808266 (joint); one-step minimax is 0.633019, 3.003574 and 3.732584 respectively. This small slice reinforces the observed conservatism; it is not a general estimate.

## Interpretation

The result suggests that **rectangularizing a single unknown prior can over-penalize adaptive plans**. If the real deployment uncertainty is a prior selected once per episode, a policy that permits the adversary to switch scenario identity at each branch solves a more pessimistic problem than the field evaluation actually poses. More Bellman depth does not automatically improve deployed performance when the uncertainty semantics are mismatched.

The next method should keep candidate-prior identity fixed across the whole episode and optimize a single contingent policy tree against its worst-case expected loss. A Pareto risk-vector frontier is a natural exact solver for a small finite candidate set. It can be more expensive, so frontier size and runtime belong in the benchmark.

## Model and limitations

- 16 joint hypotheses: chain/fork graph × three binary action-specific off-target spill parameters.
- Six strictly positive prior candidates; exact realized target-subset telemetry and three-bit outcome.
- The ambiguity set is not statistically calibrated and does not include held-out graph or mechanism families.
- Rectangular ambiguity allows branch-wise scenario switching; this is not the same as non-rectangular minimax over a persistent true prior.
- No action cost, telemetry noise/missingness, task disturbance or real-world data.
- No external benchmark was executed; no SOTA or novelty claim is made.

## Reproducibility

- Source validation/robust_bellman_ambiguity_benchmark.py: Git blob c157ab1470dc0e41aa7598724493fb3cd2a5ccfc; SHA-256 049a3f53e8974f6a6f15c46c0a785454313622fb9bdfaf0e6e8ad93dd72a7211; 15,014 bytes.
- Tests validation/test_robust_bellman_ambiguity_benchmark.py: Git blob 4d9987f82b6f85775a6d06a311310e3531dc428b; SHA-256 6f7936436b6a12db4c2c1a2d3a3faa37c107809fa0037633610a12b876bba3cf; 2,771 bytes; 9/9 dedicated tests pass.
- Full horizon 1–2 JSON: 1,296 rows, 809,596 bytes, SHA-256 336934ca410665a1b078407ec0a056f36cd65735c9579f154b2a96172100eb42.
- Horizon-three boundary JSON: 216 rows, 133,983 bytes, SHA-256 882d84ab09b147783fd664c4becc5a386cc1bfcc1253acca8ecd1f5832f5aff3.
- Combined local regression suite across eight modules at this stage: 60/60 passed. After adding the fixed-prior frontier module, combined local regression is 70/70 passed.
- Result JSON artifacts are local-only and bind to the executed source Git blob.

## Research context

Distributionally robust causal abstractions using Wasserstein ambiguity sets are established work (Felekis et al., ICML 2026). ACDB evaluates active causal discovery with budgets and separates skeleton, DAG and intervention-efficiency scores; CausalArena shows substantial ranking changes across SCM families and protocols; TCD-Arena stresses time-series discovery under 33 assumption violations. These suites motivate the next external gate, but none has been run here.

- https://proceedings.mlr.press/v306/felekis26a.html
- https://arxiv.org/abs/2609.31675
- https://arxiv.org/abs/2609.11897
- https://tcd-arena.github.io/
- https://www.nature.com/articles/s42256-023-00719-0
