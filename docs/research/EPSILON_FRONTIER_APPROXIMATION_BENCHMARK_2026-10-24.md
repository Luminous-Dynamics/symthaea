# Epsilon-bucket Pareto frontier approximation — 2026-10-24

## Objective

The fixed-prior minimax policy-tree solver is exact but suffers from Pareto-frontier growth: the six-prior horizon-two kernel objective reaches 450 non-dominated risk vectors. This experiment adds epsilon-bucket pruning and uses the exact solver as an oracle to measure the resulting **actual per-instance worst-case cross-entropy gap**.

## Method

Each risk vector is bucketed by ceil(loss / epsilon) in every candidate-prior coordinate. A deterministic representative is retained per bucket, followed by ordinary Pareto pruning. The selected approximate policy tree is then independently evaluated under each of the six fixed candidate priors and compared with the exact minimax frontier value.

This is an empirical approximation method, **not a proven horizon-independent epsilon-regret bound**. It is only acceptable as a candidate algorithm because the tested finite instances can be scored against the exact oracle.

## Full experiment

- 6 candidate true priors × 6 learner priors × 3 objectives (graph, intervention-kernel, joint) × 5 epsilon settings × horizons 1–2 = **180 rows**.
- Epsilon values: 0, 0.005, 0.01, 0.025 and 0.05 bits.
- All results are exact expected risks integrated over finite observation branches, not Monte Carlo estimates.
- Exact and approximate selected policy trees are re-scored by the same terminal cross-entropy metric under each original fixed candidate prior.

### Frontier reduction and observed gap

The kernel objective is the bottleneck. At horizon two, its maximum internal frontier size and worst-case regret gap were:

| Epsilon (bits) | Maximum kernel frontier | Reduction from exact | Mean kernel gap (bits) | Maximum kernel gap (bits) |
|---:|---:|---:|---:|---:|
| 0 (exact) | 450 | 0% | 0 | 0 |
| 0.005 | 287 | 36.2% | 0 | 0 |
| 0.01 | 131 | 70.9% | 0 | 0 |
| 0.025 | 38 | 91.6% | 0.002654 | 0.005307 |
| 0.05 | 19 | 95.8% | 0.002654 | 0.005307 |

Across the full 180-row run, epsilon 0.005 and 0.01 introduced **no observed worst-case minimax gap** compared with the exact frontier. This does not imply that all omitted Pareto vectors are dominated or that future instances will also have zero gap: it is an empirical result for this fixed model/prior grid.

At epsilon 0.025, only the kernel objective had nonzero observed gap: mean 0.002654 bits, maximum 0.005307 bits. At epsilon 0.05, the joint objective also lost some worst-case quality: mean gap 0.005684 bits and maximum 0.018871 bits. Graph objective maximum gap remained zero at all tested settings.

The approximate frontier's maximum internal size at epsilon 0.05 falls to 19 for kernel (from 450), and to 5 for joint (from 12). The graph frontier is already small. Epsilon pruning has little reason to affect the number of memoized states in this implementation; its measured effect is on the number of risk vectors retained and the associated frontier operations.

## What this means

A useful engineering point emerges: **small measured policy-value gaps may be enough to shrink the expensive frontier substantially**. On the current 16-hypothesis game, epsilon 0.01 retains 29.1% of the maximum kernel frontier while the exact re-evaluator found no change in worst-case score on the tested learner priors and horizons.

Do not interpret this as SOTA or a general guarantee. The epsilon grid was tested on six prior candidates and chain/fork graphs only; all observations include exact realized target-subset telemetry. There are no noisy/missing telemetry channels, physical intervention costs, task disturbances, or held-out graph/mechanism families. Runtime timings are recorded but not used as a speedup claim here because the exact-reference and approximate-search paths measure different phases.

## Next scientific gate

1. Derive a valid approximation guarantee for the risk-vector representation, including how local rounding error propagates through observation branches and horizon.
2. Add held-out prior mixtures and new graph/mechanism families; record exact-oracle gaps where exact search remains tractable and lower/upper bounds elsewhere.
3. Scale candidate-prior count and horizons; report wall time, peak memory, bucket counts, internal and root frontier sizes, and the frequency of nonzero gap.
4. Add action costs, telemetry corruption/missingness and disturbance; measure both structural recovery (skeleton/directed F1, SHD/SID) and proper score/calibration.
5. Compare the same intervention budgets against published active causal-discovery and robust-planning baselines.

## Prior art and external benchmarks

Epsilon-Pareto approximation and multi-objective planning are established methods, so bucketing by itself is not a novelty claim. Multi-objective RL/planning literature discusses epsilon-approximate Pareto sets (e.g. Pirotta, Parisi and Restelli, AAAI 2015: https://ojs.aaai.org/index.php/AAAI/article/view/9617). Robust causal abstractions using Wasserstein ambiguity sets have also been studied; Felekis et al. (ICML 2026) report theoretical worst-case abstraction-error guarantees in their setting: https://proceedings.mlr.press/v306/felekis26a.html.

ACDB separates skeleton recovery, directed-DAG recovery and intervention efficiency under budgeted interventions: https://arxiv.org/abs/2609.31675. CausalArena varies SCM families and protocols and reports ranking shifts across regimes: https://arxiv.org/abs/2609.11897. TCD-Arena stresses time-series causal discovery under 33 assumption violations: https://tcd-arena.github.io/. Those are the appropriate next external qualification gates; none of these benchmarks was run in the present lane, and their F1/SHD values are not numerically comparable to these cross-entropy results.

## Reproducibility

- Source: validation/epsilon_frontier_approximation_benchmark.py; Git blob 4cc98f857060d597f34524d2352fc6e0942c3089; SHA-256 6b633239eb0b8db4fc4a31cb55fb1b397e860353e9454b1842a0012dd1b2e368; 14,711 bytes.
- Tests: validation/test_epsilon_frontier_approximation_benchmark.py; Git blob 183f56efe3a77e3e837bb2059e2e62e11eff15de; SHA-256 7600169767332a064d1f923e262b15048fdc9a07e1b182bf05dd147cb9157fd7; 3,049 bytes; 9/9 dedicated tests pass.
- Full result artifact: 180 rows, 204,196 bytes, SHA-256 28b86f9dcbdf8adf233569b834ac1e7a57833f9a095ff928ca4eb85a098c9364; output is bound to source Git blob 4cc98f857060d597f34524d2352fc6e0942c3089.
- Combined local regression: **70/70 tests passed** across nine modules on Python 3.13.5.
- Exact JSON is local-only. No external benchmark run, novelty claim, or SOTA claim is made.
