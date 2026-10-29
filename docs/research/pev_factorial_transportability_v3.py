#!/usr/bin/env python3
# PEV factorial transportability benchmark v3

**Status:** synthetic reference benchmark only. The script does not import, compile, or execute production Symthaea.

## Why this increment

Factorial v1 tested whether the joint arm identifies PE/context interaction. This increment probes three ways a statistically tidy factorial estimate can still mislead:

1. **Pseudoreplication:** multiple replay blocks share a scenario/episode-level cause. Treating every row as independent makes confidence intervals too narrow.
2. **Heterogeneous treatment effects:** two contexts can have strong, opposite interaction effects that cancel in the population mean. A pooled estimate near zero must not be interpreted as “PE never matters.”
3. **Replay-order drift:** the order in which arms execute can be confounded with the intervention. A deterministic sequence plus nonlinear drift can manufacture a causal interaction where none exists.

The central change is to make the estimand and its sampling unit explicit: cluster-aware intervals for clustered trials, subgroup estimates alongside pooled effects, and randomized/counterbalanced intervention order when drift is plausible.

## Exact reproduction

```bash
python3 -m py_compile pev_factorial_transportability_v3.py
python3 pev_factorial_transportability_v3.py \
  --seeds 1000 --clusters 40 --blocks-per-cluster 8 \
  --drift-blocks 512 --base-seed 20261029 \
  --output pev_factorial_transportability_v3_results.json
```

This run used only Python’s standard library and completed 1,000 independent seeds per calibration family.

## Results

### 1. Cluster dependence

Each seed had 40 clusters × 8 replay blocks. A true interaction of +0.12 varied by a cluster-level random effect (SD 0.05); within-cluster residual SD was 0.03.

| Interval estimator | Empirical coverage of nominal 95% interval | Mean interval width |
|---|---:|---:|
| Naive row-IID t interval | 57.6% | 0.01267 |
| Cluster-mean t interval | 94.3% | 0.03246 |

This is the strongest result in the panel. The naive estimator’s narrower interval is false precision: it effectively counts correlated blocks as independent evidence. The cluster-aware interval lands in the predeclared 93–97% calibration band for this generator.

### 2. Opposing subgroup interactions

Two equally weighted contexts had true interaction +0.12 and −0.12. The pooled estimand was zero by construction. Across 1,000 seeds:

- both subgroup signs recovered: **100.0%**;
- positive subgroup sign recovered: **100.0%**;
- negative subgroup sign recovered: **100.0%**;
- pooled interaction practical-equivalence gate (±0.02): **100.0%**.

Median estimates were +0.12001, −0.12004, and +0.00006 pooled. This is not a paradox: the average effect is practically zero while the effect is strongly context-dependent. Reporting only the pooled mean would conceal the mechanism that matters for decisions.

### 3. Replay-order drift

The true interaction was zero, while arm observations received quadratic position drift with coefficient 0.02. Under a fixed order `[Y00, Y10, Y01, Y11]`, the factorial contrast acquired a planted +0.08 artifact. Across 1,000 seeds × 512 blocks:

| Replay design | Mean estimated interaction | Nominal 95% CI includes true zero |
|---|---:|---:|
| Fixed arm order | +0.08016 | 0.0% |
| Uniform randomized arm order per block | +0.00026 | 94.7% |

Randomization removes the average arm/order confounding in this fixture. It does not eliminate every kind of nonstationarity; real runs must also log order, epochs, state snapshots, and exogenous randomness.

## Gates

All 6/6 declared gates passed:

- cluster-aware interval coverage in 93–97%;
- naive IID interval exposes material cluster undercoverage;
- pooled equivalence and opposite subgroup sign recovery both hold;
- randomized ordering removes the planted drift bias;
- randomized-order interval coverage is in 93–97%;
- the missing-joint-arm, opposing-subgroup-schema, and fixed-order-drift mutations are detected.

The fixed-order false-positive failure is intentional and informative: it shows that **a pass/fail benchmark must include cases where the causal estimator should fail under a confounded design**, then verify that a design correction repairs it.

## Relation to current research and industry

This benchmark is complementary to existing agent-failure attribution methods, not a direct head-to-head result:

- [Causal Agent Replay (CAR)](https://arxiv.org/abs/2606.08275) uses forward intervention replay and budget-bounded Shapley attribution, including planted interactions.
- [REFLECT](https://arxiv.org/abs/2606.09071) uses diagnosis-specific counterfactual patches and replay outcome changes to refine error attribution.
- [GCJR / Minimal Repair Family Recovery](https://arxiv.org/abs/2608.29228) targets jointly necessary event sets; its controlled study reports 1.000 family exact match in-scope and 55.1% fewer replay calls than exhaustive search for its tested graphs.
- [DCFA (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1887/) combines dependency-graph and local counterfactual reasoning and reports up to +8.27 percentage points step-level accuracy on Who&When.
- [TraceTrial-Bench](https://github.com/TraceLite-AI/TraceTrial-Bench) evaluates harness reliability separately from task success; the published summary reports 31.1% of runs succeeding at the task despite at least one harness criterion failing.

Those numbers use different tasks, metrics, baselines and replay budgets. They cannot be used to claim that this synthetic estimator is better than any of those systems. A credible comparison would run every method on the same held-out traces and report interaction recovery, minimal repair recovery, calibration, abstentions, utility restored, replay/model calls, latency and compute cost.

## Proposed acceptance targets for the next phase

1. Across each predeclared dependence regime, target 93–97% empirical coverage for nominal 95% intervals over at least 1,000 seeds, with binomial confidence intervals.
2. Recover at least 95% of subgroup interaction signs for `|gamma_group| >= 0.10`, while correctly exposing sign reversals and not treating pooled cancellation as homogeneous null behavior.
3. For randomized/counterbalanced order, keep false-positive interaction claims at or below 5% in null drift controls; fail the fixed-order positive-bias control by design.
4. Abstain with `interaction_not_identifiable` if any required factorial arm, trace identity, or replay snapshot is absent. Never substitute zero for missing evidence.
5. On a common held-out real trace suite, compare against additive/no-interaction forecast, last-action and LLM-judge heuristics, CAR, REFLECT, GCJR and DCFA where interfaces permit. Cap replay calls and report total cost, not just accuracy.
6. Only claim production impact after exact-head Rust traces, reset/checkpoint replay equivalence, immutable evidence IDs, independent outcome scoring, and paired held-out task-utility gains are available.

## Limitations

- All numbers are generated by synthetic data-generating processes and depend on the declared cluster/noise/drift model.
- The interval calibration is evidence about these specific generators, not a universal coverage guarantee.
- No production Rust code, actual cognitive PE traces, task utility, or competitor implementation is tested.
- The interval gates are calibration targets, not SOTA claims.